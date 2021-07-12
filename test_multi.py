import os, time, sys
from glob import glob
import pathlib
import argparse
import shutil
import random
import numpy as np

# import monai
# from monai.data import PersistentDataset
# from monai import transforms as mt
# from monai.utils import set_determinism

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from PersistentDataset import PersistentDataset

from model.unet import ResNetUNet
# from model.TernausNet import UNet11, LinkNet34, UNet, UNet16, AlbuNet
# from model.UNet_3Plus import UNet_3Plus
# from model.network import R2U_Net, AttU_Net, R2AttU_Net

from collections import defaultdict
from loss import dice_loss

import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


######################################################

# moddel_list = {'UNet11': UNet11,
#                'UNet16': UNet16,
#                'UNet': UNet,
#                'AlbuNet': AlbuNet,
#                'LinkNet34': LinkNet34}

pjoin = os.path.join

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
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

    print("{}: {}".format(phase, ", ".join(outputs)))


def get_transforms(args):

    val_trans = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    return val_trans

######################################################

def validation(args, model, device, val_loader):
    metrics = defaultdict(float)
    epoch_samples = 0

    model.eval()
    
    with torch.no_grad():
        metric_count = 0
        metric_sum = 0.0

        for img_idx, val_data in enumerate(val_loader):
            
            s_val_load1 = time.time()
            val_images = val_data['img'].to(device, dtype=torch.float32)
            s_val_load2 = time.time()

            val_labels = val_data['seg'].to(device, dtype=torch.float32)
            print("================= Load img %i: %f ms" % (img_idx+1, (s_val_load2-s_val_load1)*1000))

            s1_val = time.time()
            val_outputs = model(val_images)
            s2_val = time.time()
            print("================= process img %i: %f ms" % (img_idx+1, (s2_val-s1_val)*1000))
            print(">>>>>>> Total %i: %f ms <<<<<<" % (img_idx+1, (s2_val-s_val_load1)*1000))

            loss, iou = calc_loss(val_outputs, val_labels, metrics)
            epoch_samples += val_images.size(0)

            # ---------------------------------
            ## Plot
            val_outputs = torch.sigmoid(val_outputs)
            for i in range(val_labels.shape[0]):
                for j in range(1, val_labels.shape[1]):
                    val_labels[i,j,:,:] += val_labels[i,0,:,:]*0.5
                    val_outputs[i,j,:,:] += val_outputs[i,0,:,:]*0.5

            fldr = "plot_output"
            try:
                os.makedirs(fldr, exist_ok=True)
            except TypeError:
                raise Exception("Direction not create!")
            f, axarr = plt.subplots(3)
            axarr[0].imshow(reverse_transform(val_images.squeeze_().cpu()))
            axarr[1].imshow(reverse_transform(val_labels.squeeze_()[1:4, :, :].cpu()))
            axarr[2].imshow(reverse_transform(val_outputs.squeeze_()[1:4, :, :].cpu()))
            plt.savefig(fldr+"/val_"+str(img_idx)+'.png')
            # ---------------------------------

        print_metrics(metrics, epoch_samples, 'val')
        epoch_loss = metrics['loss'] / epoch_samples
        epoch_iou = metrics['iou'] / epoch_samples

    return epoch_loss, epoch_iou

######################################################

def main(args):

    ###################################
    #         Path
    ###################################

    # config = wandb.config
    
    data_folder = './data'
    tmp_path = './tmp'

    shutil.rmtree(tmp_path, ignore_errors=True)
    persistent_cache = pathlib.Path(tmp_path, "persistent_cache")
    persistent_cache.mkdir(parents=True, exist_ok=True)
    # set_determinism(seed=0)

    ###################################
    #         Dataset
    ###################################

    images = sorted(glob(pjoin(data_folder, 'images', '*.npz')))
    segs = sorted(glob(pjoin(data_folder, 'masks', '*.npz')))

    data_dicts = [
        {"img": image_name, "seg": label_name}
        for image_name, label_name in zip(images, segs)
    ]

    print(data_dicts)

    # random.shuffle(data_dicts)

    # val_idx = int(0.2*len(images))
    val_idx = int(len(images))
    val_files = data_dicts[-val_idx:]

    val_trans = get_transforms(args)

    # # train_ds = PersistentDataset(data=train_files, transform=train_trans, cache_dir=persistent_cache)
    val_ds = PersistentDataset(data=val_files, transform=None, cache_dir=persistent_cache)

    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size, num_workers=4)#, pin_memory=torch.cuda.is_available())



    ###################################
    #         Test
    ###################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(n_class=args.num_classes)

    # if args.model == 'ResNetUNet':
    #     model = ResNetUNet(n_class=args.num_classes)
    # elif args.model == 'UNet3+':
    #     model = UNet_3Plus(n_classes=args.num_classes)
    # elif args.model =='R2U_Net':
    #     model = R2U_Net(img_ch=3, output_ch=args.num_classes, t=3)
    # elif args.model =='AttU_Net':
    #     model = AttU_Net(img_ch=3, output_ch=args.num_classes)
    # elif args.model == 'R2AttU_Net':
    #     model = R2AttU_Net(img_ch=3, output_ch=args.num_classes, t=3)
    # elif args.model == 'UNet':
    #     model = UNet(num_classes=args.num_classes)
    # else:
    #     model_name = moddel_list[args.model]
    #     model = model_name(num_classes=args.num_classes, pretrained=False)

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


    model_path = os.path.join(args.save_dir, args.load_model_name)
    
    model.load_state_dict(torch.load(model_path))  # admm train need basline model
    print("Load Model!!", args.load_model_name)                
    model.cuda()

    epoch_loss, epoch_iou = validation(args, model, device, val_loader)

    # # --------------------------------
    # print("+++++++++++++++++++++++++++++")
    # from ptflops import get_model_complexity_info
    # ### modified flops_counter.py for zero weight [conv_flops_counter_hook() & linear_flops_counter_hook()]
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=False)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print("+++++++++++++++++++++++++++++")
    # # --------------------------------

# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("--dims", default=(224, 224), type=list)
    parser.add_argument("--model", default='ResNetUNet', type=str,
                        help='[ResNetUNet, UNet, UNet11, UNet16, AlbuNet, LinkNet34]')
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument('--save_dir', type=str, default="./ckpt", metavar='N', help='Directory to save checkpoints')
    parser.add_argument('--load_model_name', type=str, default="ResNetUNet_47.pt", metavar='N', help='Model name')

    args = parser.parse_args()
    # print(args)
    
    ###################################

    args, unknown = parser.parse_known_args()
    # wandb.init(config=args)
    # wandb.config.update(args)
    
    main(args)
