#######################################################################
# Script to train the segmentation model, one model for multi labels  #
#######################################################################

import logging
import os
import sys
from glob import glob
import pathlib
import argparse
import shutil
import random
import numpy as np
import copy

from PIL import Image

import monai
from monai.data import PersistentDataset
from monai import transforms as mt
from monai.utils import set_determinism

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from model.unet import ResNetUNet
from model.TernausNet import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from model.UNet_3Plus import UNet_3Plus
from model.network import R2U_Net, AttU_Net, R2AttU_Net

from collections import defaultdict
from loss import dice_loss

import matplotlib.pyplot as plt

import wandb
wandb.init(project='Medical-CT', entity='zhoushanglin100')


######################################################

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34}

pjoin = os.path.join

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)

    iou, dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['iou'] += iou.data.cpu().numpy() * target.size(0)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

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
            mt.AddChannelD(keys=['img']),
            mt.RandGaussianNoiseD(keys=['img']),
            # mt.RandSpatialCropSamplesD(keys=['img', 'seg'], num_samples=5),
            # mt.CropForegroundD(keys=["img", "seg"], source_key="img"),
            # mt.RandGaussianSharpenD(keys=['img']),
            mt.RandShiftIntensityD(keys=['img'], offsets = 5),
            # mt.RandAdjustContrastD(keys=['img']),
            mt.ResizeD(keys=['img', 'seg'], spatial_size=tuple(args.dims)),
            # mt.RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", 
            #                             spatial_size=[224, 224], pos=1, neg=1, num_samples=4),
            # mt.RandRotate90d(keys=["img", "seg"], prob=0.3, spatial_axes=[0, 1]),
            # mt.RandHistogramShiftd(keys=["img", "seg"]),
            mt.ToTensorD(keys=['img', 'seg']),
            mt.AsDiscreteD(keys=['seg'], threshold_values=True),
        ]
    )
    val_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img']),
            mt.AddChannelD(keys=['img']),
            mt.ResizeD(keys=['img', 'seg'], spatial_size=tuple(args.dims)),
            # mt.CropForegroundd(keys=["img", "seg"], source_key="img"),
            mt.ToTensorD(keys=['img', 'seg']),
            mt.AsDiscreteD(keys=['seg'], threshold_values=True),
        ]
    )
    
    return train_trans, val_trans



######################################################

def train(args, model, device, train_loader, optimizer):#, scheduler):
    model.train()

    metrics = defaultdict(float)
    epoch_samples = 0
    for batch_data in train_loader:
        inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)
                
        optimizer.zero_grad()

        print("***********")
        print(inputs.shape)
        print("***********")

        outputs = model(inputs)
        loss, iou = calc_loss(outputs, labels, metrics)
        loss.backward()
        optimizer.step()

        epoch_samples += inputs.size(0)
    
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss'] / epoch_samples
    epoch_iou = metrics['iou'] / epoch_samples
    
    # scheduler.step()
    wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})




def validation(args, model, device, val_loader):
    metrics = defaultdict(float)
    epoch_samples = 0

    model.eval()
    
    with torch.no_grad():
        metric_count = 0
        metric_sum = 0.0
        for val_data in val_loader:
            val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)
            
            val_outputs = model(val_images)

            loss, iou = calc_loss(val_outputs, val_labels, metrics)
            epoch_samples += val_images.size(0)

            # -------------
            ## Plot
            val_outputs = F.sigmoid(val_outputs)
            for i in range(val_labels.shape[0]):
                for j in range(1, val_labels.shape[1]):
                    val_labels[i,j,:,:] += val_labels[i,0,:,:]*0.5
                    val_outputs[i,j,:,:] += val_outputs[i,0,:,:]*0.5

            wandb.log({"masks/Image": [wandb.Image(val_images, caption="Images")]})
            wandb.log({"masks/true": [wandb.Image(reverse_transform(val_labels.squeeze_()[1:4, :, :].cpu()), caption="Mask/True")]})
            wandb.log({"masks/pred": [wandb.Image(reverse_transform(val_outputs.squeeze_()[1:4, :, :].cpu()), caption="Mask/Pred")]})
            # ---------------------------------

        print_metrics(metrics, epoch_samples, 'val')
        epoch_loss = metrics['loss'] / epoch_samples
        epoch_iou = metrics['iou'] / epoch_samples

    return epoch_loss, epoch_iou, model.state_dict()


def main(args):

    ###################################
    #         Path
    ###################################

    config = wandb.config
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_folder = './data'
    image_folder = 'img_'+args.data_view
    mask_folder = 'mask_'+args.data_view
    tmp_path = './tmp'

    shutil.rmtree(tmp_path, ignore_errors=True)
    persistent_cache = pathlib.Path(tmp_path, "persistent_cache")
    persistent_cache.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=0)

    ###################################
    #         Dataset
    ###################################

    # images = sorted(glob(pjoin(data_folder, 'train_polar', 'images_polar_2', '*.npz')))
    # segs = sorted(glob(pjoin(data_folder, 'train_polar', 'masks_polar_2', '*.npz')))

    images = sorted(glob(pjoin(data_folder, image_folder, '*.npz')))
    segs = sorted(glob(pjoin(data_folder, mask_folder, '*.npz')))

    # aa = sorted(glob(pjoin('../Medical/data', 'train', 'images', '*.npz')))

    data_dicts = [
        {"img": image_name, "seg": label_name}
        for image_name, label_name in zip(images, segs)
    ]

    random.shuffle(data_dicts)

    val_idx = int(0.1*len(images))

    train_files, val_files = data_dicts[:-val_idx], data_dicts[-val_idx:]
    train_trans, val_trans = get_transforms(args)

    train_ds = PersistentDataset(data=train_files, transform=train_trans, cache_dir=persistent_cache)
    val_ds = PersistentDataset(data=val_files, transform=val_trans, cache_dir=persistent_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size, num_workers=4)#, pin_memory=torch.cuda.is_available())

    ###################################
    #         Train
    ###################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ResNetUNet(n_class=4).to(device)

    if args.model == 'ResNetUNet':
        model = ResNetUNet(n_class=args.num_classes)

    elif args.model == 'UNet3+':
        model = UNet_3Plus(n_classes=args.num_classes)


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

    # # ----------------------------
    # from torchsummary import summary
    # from prettytable import PrettyTable

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
    # # --------------------------------

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = 1e10
    best_iou = 0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 50)

        train(args, model, device, train_loader, optimizer)#, scheduler)

        if (epoch + 1) % args.val_inter == 0:

            epoch_loss, epoch_iou, best_state = validation(args, model, device, val_loader)

            if epoch_iou > best_iou:
                print("Get a new best test iou:{:.2f}\n".format(epoch_iou))
                best_iou = epoch_iou
                best_metric_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

                model_name = "checkpoints/ckpt/"+args.data_view+"_"+args.model+"_"+str(epoch)+".pt"
                torch.save(best_model_wts, model_name)

            print("current epoch: {}; current test iou: {:.4f}; best test iou: {:.4f} at epoch {}\n".format(
                    epoch+1, epoch_iou, best_iou, best_metric_epoch))
            
        
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("--dims", default=(224, 512), type=list, 
    #                     help="Original shape: [(512, 512) -> (512, 512), (221, 512) -> (224, 512)]")
    parser.add_argument("--dims", default=(224, 512), nargs='+', type=int, 
                        help="Original shape: [(512, 512) -> (512, 512), (221, 512) -> (224, 512)]")
    parser.add_argument("--model", default='ResNetUNet', type=str,
                        help='Choose from [ResNetUNet, UNet, UNet11, UNet16, AlbuNet, LinkNet34]')
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument('--data_view', type=str, default="updown", metavar='N',
                        help="Choose from ['updown', 'leftright', 'frontback']")
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    args = parser.parse_args()
    print(args)
    
    ###################################

    args, unknown = parser.parse_known_args()
    wandb.init(config=args)
    wandb.config.update(args)
    
    main(args)
