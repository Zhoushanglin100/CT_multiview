#######################################################################
# Script to train the segmentation model, one model for single labels #
#######################################################################


import logging
import os
import sys
from glob import glob
import pathlib
import argparse
import shutil
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader

import monai
from monai.data import ArrayDataset
from monai.data import PersistentDataset, list_data_collate, SmartCacheDataset, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai import transforms as mt
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
from torch.nn.functional import interpolate

from monai.optimizers import Novograd

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from model_bank import model_bank
from monai.networks.nets import DynUNet, UNet
from monai.visualize import plot_2d_or_3d_image

import matplotlib.pyplot as plt

import wandb
wandb.init(project='Medical-CT', entity='zhoushanglin100')

######################################################

pjoin = os.path.join

def compute_loss_list(loss_fn, preds, label):
    labels = [label] + [
        interpolate(label, pred.shape[2:]) for pred in preds[1:]
    ]
    return sum(
        0.5 ** i * loss_fn(p, l)
        for i, (p, l) in enumerate(zip(preds, labels))
    )

def get_transforms(args):
    train_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img']),
            mt.AddChannelD(keys=['img', 'seg']),
            mt.RandGaussianNoiseD(keys=['img']),
            # mt.RandGaussianSharpenD(keys=['img']),
            # mt.RandShiftIntensityD(keys=['img'], offsets = 5),
            # mt.RandAdjustContrastD(keys=['img']),
            # mt.ResizeD(keys=['img', 'seg'], spatial_size=args.dims),
            # mt.RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", 
            #                             spatial_size=[96, 96], pos=1, neg=1, num_samples=4),
            # mt.RandRotate90d(keys=["img", "seg"], prob=0.8, spatial_axes=[0, 1]),
            # mt.RandHistogramShiftd(keys=["img", "seg"]),
            mt.ToTensorD(keys=['img', 'seg']),
            mt.AsDiscreteD(keys=['seg'], threshold_values=True),
        ]
    )
    val_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img']),
            mt.AddChannelD(keys=['img', 'seg']),
            # mt.ResizeD(keys=['img', 'seg'], spatial_size=args.dims),
            mt.ToTensorD(keys=['img', 'seg']),
            mt.AsDiscreteD(keys=['seg'], threshold_values=True),
        ]
    )
    return train_trans, val_trans



def main_worker(args):
    ###################################
    #         Initial config
    ###################################
    # if args.dist:
    #     print(f'Starting local rank {args.local_rank}')
    #     if args.local_rank != 0:
    #         f = open(os.devnull, "w")
    #         sys.stdout = sys.stderr = f
    #     dist.init_process_group(backend="nccl", init_method="env://")

    config = wandb.config

    monai.config.print_config()
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_folder = './data'
    home_folder = '..'
    tmp_path = './tmp'

    shutil.rmtree(tmp_path, ignore_errors=True)
    persistent_cache = pathlib.Path(tmp_path, "persistent_cache")
    persistent_cache.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=0)


    images = sorted(glob(pjoin(data_folder, 'train', 'images', '*.npz')))
    segs = sorted(glob(pjoin(data_folder, 'train', 'masks', '*.npz')))

    data_dicts = [
        {"img": image_name, "seg": label_name}
        for image_name, label_name in zip(images, segs)
    ]

    random.shuffle(data_dicts)

    ###################################
    #         Dataset
    ###################################

    val_idx = int(0.1*len(images))

    train_files, val_files = data_dicts[:-val_idx], data_dicts[-val_idx:]
    train_trans, val_trans = get_transforms(args)


    # if args.dist:
    #     train_data_part = partition_dataset(
    #         data=train_files,
    #         num_partitions=dist.get_world_size(),
    #         shuffle=True,
    #         even_divisible=True,
    #     )[dist.get_rank()]
    #     val_data_part = partition_dataset(
    #         data=val_files,
    #         num_partitions=dist.get_world_size(),
    #         shuffle=False,
    #         even_divisible=True,
    #     )[dist.get_rank()]


    train_ds = PersistentDataset(data=train_files, transform=train_trans, cache_dir=persistent_cache)
    val_ds = PersistentDataset(data=val_files, transform=val_trans, cache_dir=persistent_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size, num_workers=4)#, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = mt.Compose([
        mt.Activations(sigmoid=True),
        mt.AsDiscrete(threshold_values=True),
    ])

    # ----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(dimensions=2, in_channels=1, out_channels=1,
                 channels=(8, 16, 32, 64, 128, 256), 
                 strides=(2,2,2,2,2), 
                 # kernel_size=3,
                 # num_res_units=3,
                 # dropout=0.1,
                 ).to(device)
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #                         in_channels=1, out_channels=1, pretrained=False).to(device)
    # ----------------------------------------------------------------------

    # if args.dist:
    #     device = torch.device(f"cuda:{args.local_rank}")
    #     model = DDP(model_bank(args).to(device), device_ids=[device])
    # else:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = model_bank(args).to(device)

    loss_function = monai.losses.DiceLoss(sigmoid=True)
    
    if args.fast:
        optimizer = Novograd(model.parameters(), args.lr * 50)
        scaler = torch.cuda.amp.GradScaler()
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    ###################################
    #         Training
    ###################################
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    wandb.watch(model)
    # if args.dist:
    #     train_ds.start()
    #     val_ds.start()

    for epoch in range(args.epochs):
        print("-" * 50)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)
            optimizer.zero_grad()
            if args.fast:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if args.arch == 'dynunet':
                        loss = compute_loss_list(loss_function, outputs, labels)
                    else:
                        loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                # if args.arch == 'dynunet':
                #     loss = compute_loss_list(loss_function, outputs, labels)
                # else:
                loss = loss_function(outputs, labels)
                # print(torch.max(labels), torch.min(labels))
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            if step % args.print_freq == 0:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            
            wandb.log({"train_loss": loss})
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        ###################################
        #         Validation
        ###################################

        if (epoch + 1) % args.val_inter == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)

                    # if args.fast:
                    #     with torch.cuda.amp.autocast():
                    #         val_outputs = model(val_images)
                    # else:
                    val_outputs = model(val_images)
                    
                    # roi_size = args.dims
                    # sw_batch_size = 4
                    # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    
                    val_outputs = post_trans(val_outputs)
                    value, _ = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), pjoin('checkpoints', f'{args.arch}_best.pth'))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

                wandb.log({"val_mean_dice": metric})
                wandb.log({"Best Validation Mean Dice": best_metric})

                #### plot the last model output as GIF image in TensorBoard with the corresponding image and label
                # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

                fldr = "plot/ultra_"+args.ext
                try:
                    os.makedirs(fldr, exist_ok=True)
                except TypeError:
                    raise Exception("Direction not create!")
                f, axarr = plt.subplots(3)
                axarr[0].imshow(val_images.squeeze_().cpu().numpy())
                axarr[1].imshow(val_labels.squeeze_().cpu().numpy())
                axarr[2].imshow(val_outputs.squeeze_().cpu().numpy())
                plt.savefig(fldr+"/val_"+str(epoch)+'.png')

                # plt.imshow(val_images.squeeze_().cpu().numpy())
                # plt.savefig("plot/ultra/image_"+str(epoch)+'.png')
                
                # plt.imshow(val_labels.squeeze_().cpu().numpy())
                # plt.savefig("plot/ultra/labels_"+str(epoch)+'.png')

                # plt.imshow(val_outputs.squeeze_().cpu().numpy())
                # plt.savefig("plot/ultra/outputs_"+str(epoch)+'.png')

                # wandb.log({"examples" : [wandb.Image(i) for i in val_outputs]})

                wandb.log({"masks/Image": [wandb.Image(val_images, caption="Images")]})
                wandb.log({"masks/true": [wandb.Image(val_labels, caption="Mask/True")]})
                wandb.log({"masks/pred": [wandb.Image(val_outputs, caption="Mask/Pred")]})

            scheduler.step(metric)
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})

        # if args.dist:
        #     train_ds.update_cache()
        #     val_ds.update_cache()
    # if args.dist:
    #     train_ds.shutdown()
    #     val_ds.shutdown()
    #     if dist.get_rank() == 0:
    #         torch.save(model.state_dict(), pjoin(home_folder, 'checkpoints', f'{args.arch}_final.pth'))
    #     dist.destroy_process_group()
    # else:
    torch.save(model.state_dict(), pjoin('checkpoints', f'{args.arch}_final.pth'))

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--data", default='gd', type=str)
    # parser.add_argument("--dims", default=(224, 224), type=list)
    parser.add_argument("--arch", default='unet', type=str)
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--fast", default=False, type=bool)
    # parser.add_argument("--dist", action='store_true')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--ext", default='unet', type=str)
    args = parser.parse_args()
    print(args)
    main_worker(args)

if __name__ == "__main__":
    main()