#######################################################################
# Script for model compression: ADMM train                            #
# Tried only on ResNetUNet                                            #
#######################################################################


from __future__ import print_function

import os, sys, shutil, random, copy, pickle
from glob import glob
import pathlib
import argparse
from torchsummary import summary
from prettytable import PrettyTable
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import monai
from monai.data import PersistentDataset
from monai import transforms as mt
from monai.utils import set_determinism

from model.unet import ResNetUNet
from model.TernausNet import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from model.UNet_3Plus import UNet_3Plus
from model.network import R2U_Net, AttU_Net, R2AttU_Net

from collections import defaultdict
from loss import dice_loss

import admm.admm as admm
# from admm.parameters import *

import matplotlib.pyplot as plt
#matplotlib.use("agg")

import logging
LOG_FILENAME = 'output.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

import wandb
wandb.init(project='Medical-CT', entity='zhoushanglin100')

writer = None

torch.cuda.empty_cache()

########################################################################

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34}

pjoin = os.path.join


def save_ckp(args, state, is_best, checkpoint_dir, best_model_dir, model_name):
    f_path = checkpoint_dir+"/"+model_name
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir+"/"+model_name
        shutil.copyfile(f_path, best_fpath)

def load_ckp(args, checkpoint_fpath, model, optimizer):

    model_name = checkpoint_fpath+"/{}_admmtrain_{}_{}{}.pt".format(args.data_view, args.prun_config_file, args.sparsity_type, args.ext)
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer, checkpoint['epoch']


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

def calc_loss(pred, target, metrics, phase, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    iou, dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['iou'] += iou.data.cpu().numpy() * target.size(0)
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    wandb.log({"Loss/"+phase+"_iter_loss": loss})
    wandb.log({"IOU/"+phase+"_iter_iou": iou})

    return loss, iou


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        if k == "iou":
            wandb.log({"IOU/"+phase+"_"+k: metrics[k]/epoch_samples})
        else:
            wandb.log({"Loss/"+phase+"_"+k: metrics[k]/epoch_samples})

    print("{}: {}".format(phase, ", ".join(outputs)))


def get_transforms(args):
    train_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img']),
            mt.AddChannelD(keys=['seg']),
            mt.RandGaussianNoiseD(keys=['img']),
            mt.RandShiftIntensityD(keys=['img'], offsets = 5),
            mt.ResizeD(keys=['img', 'seg'], spatial_size=(args.dims, args.dims)),
            mt.ToTensorD(keys=['img', 'seg']),
            mt.AsDiscreteD(keys=['seg'], threshold_values=True),
        ]
    )
    val_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img']),
            mt.AddChannelD(keys=['seg']),
            mt.ResizeD(keys=['img', 'seg'], spatial_size=(args.dims, args.dims)),
            mt.ToTensorD(keys=['img', 'seg']),
            mt.AsDiscreteD(keys=['seg'], threshold_values=True),
        ]
    )
    
    return train_trans, val_trans


def total_params(model):
        return sum([np.prod(param.size()) for param in model.parameters()])

def param_to_array(param):
    return param.cpu().data.numpy().reshape(-1)

def get_sorted_list_of_params(model):
    params = list(model.parameters())
    param_arrays = [param_to_array(param) for param in params]
    return np.sort(np.concatenate(param_arrays))

########################################################################

def train(args, ADMM, model, device, train_loader, optimizer, epoch, writer):
    model.train()

    ce_loss = None
    mixed_loss = None
    ctr=0
    total_ce = 0
    
    metrics = defaultdict(float)
    epoch_samples = 0

    for batch_idx, batch_data in enumerate(train_loader):

        inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)

        ctr += 1

        optimizer.zero_grad()
        outputs = model(inputs)
     
        dice_loss, iou = calc_loss(outputs, labels, metrics, 'train')
        total_ce = total_ce + float(dice_loss.item())
        
        admm.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, inputs, batch_idx, writer)  # update Z and U variables

        dice_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, dice_loss)  # append admm losss

        wandb.log({"Loss/train_itr_mixed_loss": mixed_loss})

        mixed_loss.backward(retain_graph=True)
        optimizer.step()
 
        epoch_samples += inputs.size(0)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, iou: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), dice_loss.item(), iou))
            print("Cross_entropy loss: {}, mixed_loss : {}".format(dice_loss, mixed_loss))

    wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})

    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss'] / epoch_samples
    epoch_iou = metrics['iou'] / epoch_samples
  
    lossadmm = []
    for k, v in admm_loss.items():
        # print("at layer {}, admm loss is {}".format(k, v))
        lossadmm.append(float(v))
 
    if args.verbose:
        for k, v in admm_loss.items():
            # print("at layer {}, admm loss is {}".format(k, v))
            ADMM.admmloss[k].extend([float(v)])

    ADMM.ce_prev = ADMM.ce
    ADMM.ce = total_ce / ctr


# ---------------------------------------------------------------------

def masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch):

    if not args.masked_retrain:
        return

    idx_loss_dict = {}
    
    model.train()

    masks = {}

    metrics = defaultdict(float)
    epoch_samples = 0

    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        above_threshold, W = admm.weight_pruning(args, W, ADMM.prune_ratios[name])
        W.data = W
        masks[name] = above_threshold

    for batch_idx, batch_data in enumerate(train_loader):

        inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss, iou = calc_loss(outputs, labels, metrics, 'retrain')
        loss.backward()

        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]
        
        optimizer.step()
        epoch_samples += inputs.size(0)

        if batch_idx % args.log_interval == 0:

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            print('Retrain Epoch: {} [{}/{} ({:.0f}%)] [{}]\tLoss: {:.6f}, iou: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), current_lr, loss.item(), iou))
            print("({}) cross_entropy loss: {}".format(args.sparsity_type, loss))


        if batch_idx % 1 == 0:
            idx_loss_dict[batch_idx] = loss.item()
    

    wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
    print_metrics(metrics, epoch_samples, 'retrain')
    epoch_loss = metrics['loss'] / epoch_samples
    epoch_iou = metrics['iou'] / epoch_samples

    return idx_loss_dict

# ---------------------------------------------------------------------


def validation(args, model, device, val_loader):
    metrics = defaultdict(float)
    epoch_samples = 0

    model.eval()

    with torch.no_grad():

        # for val_data in val_loader:
        for batch_idx, val_data in enumerate(val_loader):

            val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)

            val_outputs = model(val_images)

            val_loss, val_iou = calc_loss(val_outputs, val_labels, metrics, 'val')
            
            epoch_samples += val_images.size(0)

            # # -------------
            # ## Plot
            # if batch_idx % args.plot_interval == 0:
            #     val_outputs[0] = F.sigmoid(val_outputs[0])

            #     wandb.log({"masks/Image": [wandb.Image(val_images[0], caption="Images")]})
            #     wandb.log({"masks/true": [wandb.Image(val_labels[0].squeeze_().cpu(), caption="Mask/True")]})
            #     wandb.log({"masks/pred": [wandb.Image(val_outputs[0].squeeze_().cpu(), caption="Mask/Pred")]})
            #     # wandb.log({"masks/true": [wandb.Image(reverse_transform(val_labels.squeeze_().cpu()), caption="Mask/True")]})
            #     # wandb.log({"masks/pred": [wandb.Image(reverse_transform(val_outputs.squeeze_().cpu()), caption="Mask/Pred")]})
            # # ---------------------------------

            if batch_idx % args.log_interval == 0:
                print('Validation: [{}/{} ({:.0f}%)]\tLoss: {:.6f}, iou: {:.6f}'.format(
                       batch_idx * len(val_images), len(val_loader.dataset),
                            100. * batch_idx / len(val_loader), val_loss, val_iou))

        print_metrics(metrics, epoch_samples, 'val')
        epoch_loss = metrics['loss'] / epoch_samples
        epoch_iou = metrics['iou'] / epoch_samples

    return epoch_loss, epoch_iou


########################################################################

def main(args):


    ###################################
    #         Path
    ###################################

    config = wandb.config
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_folder = './data/3d'
    image_folder = 'img_'+args.data_view
    mask_folder = 'mask_'+args.data_view
    tmp_path = './tmp'

    shutil.rmtree(tmp_path, ignore_errors=True)
    # persistent_cache = pathlib.Path(tmp_path, "persistent_cache")
    # persistent_cache.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=0)

    ###################################
    #         Dataset
    ###################################

    images = sorted(glob(pjoin(data_folder, image_folder, '*.npz')))
    segs = sorted(glob(pjoin(data_folder, mask_folder, '*.npz')))

    data_dicts = [
        {"img": image_name, "seg": label_name}
        for image_name, label_name in zip(images, segs)
    ]

    random.shuffle(data_dicts)

    val_idx = int(0.1*len(images))

    train_files, val_files = data_dicts[:-val_idx], data_dicts[-val_idx:]
    train_trans, val_trans = get_transforms(args)

    # train_ds = PersistentDataset(data=train_files, transform=train_trans, cache_dir=persistent_cache)
    # val_ds = PersistentDataset(data=val_files, transform=val_trans, cache_dir=persistent_cache)

    train_ds = PersistentDataset(data=train_files, transform=train_trans, cache_dir=None)
    val_ds = PersistentDataset(data=val_files, transform=val_trans, cache_dir=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size, num_workers=16)#, pin_memory=torch.cuda.is_available())

    ###################################
    #         Model
    ###################################    
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'ResNetUNet':
        model = ResNetUNet(n_class=args.num_classes)

    elif args.model =='R2U_Net':
        model = R2U_Net(img_ch=3, output_ch=args.num_classes, t=3)
    elif args.model =='AttU_Net':
        model = AttU_Net(img_ch=3, output_ch=args.num_classes)
    elif args.model == 'R2AttU_Net':
        model = R2AttU_Net(img_ch=3, output_ch=args.num_classes, t=3)
	
    elif args.model == 'UNet':
        model = UNet(num_classes=args.num_classes)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=args.num_classes, pretrained=False)

    model = model.to(device) 

    print("\nArch name is {}".format(args.model))
    wandb.watch(model)

    # -------------------------------------------------------
    # for i, (name, W) in enumerate(model.named_parameters()):
    #     print(i, "th weight:", name, ", shape = ", W.shape, ", weight.dtype = ", W.dtype)  
    
    # summary(model, input_size=(3, 480, 640))

    # def count_parameters(model):
    #     table = PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         param = parameter.numel()
    #         table.add_row([name, param])
    #         total_params+=param
    #     print(table)
    #     print(f"Total Trainable Params: {total_params}")
    #     return total_params
        
    # count_parameters(model)
    
    # -------------------------------------------------------

    """====================="""
    """ multi-rho admm train"""
    """====================="""


    initial_rho = args.rho
    if args.admm_train:
        
        model_path = os.path.join(args.save_dir, args.data_view+"_"+args.load_model_name)
        print("!!!!!!!!!!!!", model_path)

        for i in range(args.rho_num):

            current_rho = initial_rho * 10 ** i
            if i == 0:

                model.load_state_dict(torch.load(model_path))  # admm train need basline model
                print("Pretrained model loaded!")                
                model.cuda()

                validation(args, model, device, val_loader)
            
            # ---------- Start admm train ----------
            ADMM = admm.ADMM(args, model, file_name="./profile/" + args.prun_config_file + ".yaml", rho=current_rho)
            admm.admm_initialization(args, ADMM, model)  # intialize Z and U variables

            # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.admm_train_epochs * len(train_loader), 
            #                                                     eta_min=4e-08)

            best_iou = 0
            best_metric_epoch = 1
            for epoch in range(1, args.admm_train_epochs + 1):
                print("\nepoch: ", epoch)

                # admm.admm_adjust_learning_rate(args, optimizer, epoch)
                # scheduler.step()

                train(args, ADMM, model, device, train_loader, optimizer, epoch, writer)
                epoch_loss, epoch_iou = validation(args, model, device, val_loader)
                
                if (epoch_iou > best_iou) and (epoch != 1):
                    is_best = epoch_iou > best_iou
                    print("\nGet a new best test iou:{:.2f}".format(epoch_iou))
                    best_iou = epoch_iou
                    best_metric_epoch = epoch
                    best_admm_model = model

                    model_name = "{}_admmtrain_{}_{}{}.pt".format(args.data_view, args.prun_config_file, args.sparsity_type, args.ext)
                    checkpoint_dir = "ckpt_pruned/admmtrain"
                    best_model_dir = "ckpt_pruned/admmtrain/best"

                    checkpoint = {'epoch': epoch + 1,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()
                                  }
                    save_ckp(args, checkpoint, is_best, 
                             checkpoint_dir, 
                             best_model_dir, 
                             model_name)

                    # torch.save(best_model_wts, 
                    #             "ckpt_pruned/admmtrain/{}_admmtrain_{}_{}{}.pt".format(args.data_view, args.prun_config_file, args.sparsity_type, args.ext))
                    
                    print("Save new ADMM trained model!!")

                print("Condition 1")
                print(ADMM.condition1)
                print("Condition 2")
                print(ADMM.condition2)
            
                print("Current epoch: {}; current test iou: {:.4f}; best test iou: {:.4f} at epoch {}\n".format(
                        epoch, epoch_iou, best_iou, best_metric_epoch))
            
            print("\n>>>> Sparsity after admm >>>> \n")
            admm.test_sparsity(args, ADMM, best_admm_model)
            
            print("\n>_>_>_>_>_>_>_>_>_>_>_>_>_>_>_>_ Accuracy after ADMM Train:")  
            validation(args, best_admm_model, device, val_loader)


            ### ------- Hard Prune -------
            model_forhard = best_admm_model
            ADMM = admm.ADMM(args, model_forhard, file_name="./profile/" + args.prun_config_file + ".yaml", rho=current_rho)
            admm.hard_prune(args, ADMM, model_forhard)

            print("\n>>>> Sparsity after Hard Prune >>>> \n")
            admm.test_sparsity(args, ADMM, model_forhard)

            print("\n>_>_>_>_>_>_>_>_>_>_>_>_>_>_>_>_ Accuracy after Hard Prune:")           
            validation(args, model_forhard, device, val_loader)

            torch.save(model_forhard.state_dict(), 
                        "ckpt_pruned/hardprune/{}_hardprune_{}_{}{}.pt".format(args.data_view, args.prun_config_file, args.sparsity_type, args.ext))
            
    """========================"""
    """END multi-rho admm train"""
    """========================"""


    if args.masked_retrain:

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # optimizer = optim.SGD(model.parameters(), lr=mylr, momentum=momentum)

        print("\n>>>>>>>>>>>>>> Loading ADMM trained file...")
        
        file_path = "ckpt_pruned/admmtrain/best"
        model, _, _ = load_ckp(args, file_path, model, optimizer)

        model.cuda()

        print("\n==================> Before retrain starts (accuracy of admm trained model):")
        validation(args, model, device, val_loader)

        print("\n>>>> Sparsity after Hard Prune >>>> \n")
        ADMM = admm.ADMM(args, model, file_name="profile/" + args.prun_config_file + ".yaml", rho=initial_rho)
        admm.hard_prune(args, ADMM, model)
        
        admm.test_sparsity(args, ADMM, model)

        print("\n===================> Accuracy after hard_prune:")
        validation(args, model, device, val_loader)

        
        # ----------- Maske Retrain ------------------

        best_iou = 0

        for epoch in range(1, args.retrain_epochs+1):
            print("\nepoch: ", epoch)

            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.retrain_epochs * len(train_loader),
            #                                                  eta_min=4e-08)
            # scheduler.step()

            idx_loss_dict = masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch)
            _, epoch_iou = validation(args, model, device, val_loader)

            if epoch_iou > best_iou:
                print("Get a new best test IOU:{:.2f}".format(epoch_iou))
                best_iou = epoch_iou
                best_metric_epoch = epoch + 1
                best_retrain = model

                print("Saving model...")
                torch.save(model.state_dict(), 
                            "ckpt_pruned/retrain/{}_retrain_{}_{}{}.pt".format(args.data_view, args.prun_config_file, args.sparsity_type, args.ext))
            
            print("\n==============================> Accuracy after Masked Retrain of epoch {}: ".format(epoch))           
            validation(args, model, device, val_loader)
            
            print("Current epoch: {}; current test iou: {:.4f}; best test iou: {:.4f} at epoch {}\n".format(
                    epoch+1, epoch_iou, best_iou, best_metric_epoch))

        print("\n>_>_>_>_>_>_>_>_>_>_>_>_>_ Accuracy after Retrain:")  
        validation(args, best_retrain, device, val_loader)

        print("\n>>>> Sparsity after Masked Retrain >>>> \n")
        admm.test_sparsity(args, ADMM, model)


    """=================="""
    """End masked retrain"""
    """=================="""

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dims", default=256, type=list, 
                        help="Original shape: [(512, 512) -> (512, 512), (221, 512) -> (224, 512)]")
    parser.add_argument("--model", default='ResNetUNet', type=str,
                        help='[ResNetUNet, UNet, UNet11, UNet16, AlbuNet, LinkNet34]')
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument('--data_view', type=str, default="updown", metavar='N',
                        help="Choose from ['updown', 'leftright', 'frontback']")
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_dir', type=str, default="./ckpt/best_1", metavar='N', help='Directory to save checkpoints')
    parser.add_argument('--load_model_name', type=str, default="ResNetUNet_best.pt", metavar='N', help='Model name')

    # --------------- ADMM Related args --------------------------------------------------------------
    
    parser.add_argument('--optimization', type=str, default='savlr', help='admm or savlr')
    
    ### SAVLR parameters:
    parser.add_argument('--M', type=int, default=300, metavar='N', help='')
    parser.add_argument('--r', type=int, default=0.1, metavar='N', help='')
    parser.add_argument('--initial_s', type=int, default=0.01, metavar='N', help='')

    parser.add_argument('--admm_train', action='store_true', default=False, help="for admm training")
    parser.add_argument('--masked_retrain', action='store_true', default=False, help='for masked retrain')
    parser.add_argument('--admm_epoch', type=int, default=1, help="how often we do admm update")
    parser.add_argument('--admm_train_epochs', type=int, default=80, metavar='N', help='number of epochs for admm train (default: 50)')
    parser.add_argument('--rho', type=float, default=0.1, help="define rho for ADMM")
    parser.add_argument('--rho_num', type=int, default = 1, help ="define how many rohs for ADMM training")
    parser.add_argument('--sparsity_type', type=str, default='irregular', help="define sparsity_type: [irregular,column,filter]")
    parser.add_argument('--optimizer', type=str, default='adam', help='define optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='define lr scheduler, [default, cosine]')
    parser.add_argument('--prun_config_file', type=str, default='config_ResNetUNet_0.9', help="prune config file")
    parser.add_argument('--retrain_epochs', type=int, default=10, metavar='N', help='number of epochs to retrain (default: 30)')
    parser.add_argument('--verbose', action='store_true', default=False, help="whether to report admm convergence condition")
    parser.add_argument('--ext', type=str, default="", metavar='N', help='extension for saved file')

    # -----------------------------------------------------------------------------

    args = parser.parse_args()
    print(args)
    
    ###################################

    args, unknown = parser.parse_known_args()
    wandb.init(config=args)
    wandb.config.update(args)

    main(args)