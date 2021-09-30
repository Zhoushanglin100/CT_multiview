#######################################################################
# Script to test the segmentation model, one model for multi labels  #
#######################################################################

import logging
import os, time
import sys
from glob import glob
import argparse
import random
import numpy as np
import nibabel as nib


import monai
from monai.data import PersistentDataset
from monai import transforms as mt
from monai.utils import set_determinism

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from model.unet import ResNetUNet

from collections import defaultdict
from loss import dice_loss

import matplotlib.pyplot as plt
from statistics import mean

import wandb
wandb.init(project='Medical-CT', entity='zhoushanglin100')


######################################################

pjoin = os.path.join


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


def get_transforms(args, h=256, w=256):
    val_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img']),
            mt.AddChannelD(keys=['seg']),

            mt.ResizeD(keys=['img', 'seg'], spatial_size=(args.dims, args.dims)),
            # mt.ResizeD(keys=['seg'], spatial_size=(1, args.dims, args.dims)),

            # mt.CropForegroundd(keys=["img", "seg"], source_key="img"),
            mt.ToTensorD(keys=['img', 'seg']),
            mt.AsDiscreteD(keys=['seg'], threshold_values=True),
        ]
    )

    val_trans_bk = mt.Compose(
        [
            mt.Resize(spatial_size=(h, w)),
        ]
    )

    return val_trans, val_trans_bk


######################################################

def validation(args, model, device, val_loader, val_trans_bk):
    metrics = defaultdict(float)
    epoch_samples = 0

    model.eval()
    
    one_time = []

    with torch.no_grad():
        
        val_img_d, val_img_lst = {}, []
        val_msk_d, val_msk_lst = {}, []
        val_tru_d, val_tru_lst = {}, []

        for batch_idx, val_data in enumerate(val_loader):
            print(batch_idx)

            s_val_load1 = time.time()
            val_images_name = val_data["img_meta_dict"]['filename_or_obj'][0].split("/")[-1].split(".")[0]
            # print("------------>", val_images_name)

            val_images = val_data['img'].to(device)
            s_val_load2 = time.time()

            val_labels = val_data['seg'].to(device)
            val_labels_name = val_data["seg_meta_dict"]['filename_or_obj'][0].split("/")[-1].split(".")[0]

            # print("================= Load img %i: %f ms" % (batch_idx+1, (s_val_load2-s_val_load1)*1000))

            s1_val = time.time()
            val_outputs = model(val_images)
            s2_val = time.time()
            print("================= process img %i: %f ms" % (batch_idx+1, (s2_val-s1_val)*1000))
            print(">>>>>>> Total %i: %f ms <<<<<<" % (batch_idx+1, (s2_val-s_val_load1)*1000))
            one_time.append((s2_val-s1_val)*1000)


            calc_loss(val_outputs, val_labels, metrics)
            epoch_samples += val_images.size(0)

            if (batch_idx == 0):
                val_msk_d["first"] = val_labels_name
                val_tru_d["first"] = val_labels_name
            if (batch_idx == len(val_loader)-1):
                val_msk_d["last"] = val_labels_name
                val_tru_d["last"] = val_labels_name

            # ---------------------------------
            ## Plot
            val_outputs = torch.round(torch.sigmoid(val_outputs))

            # val_outputs_bk = val_trans_bk(val_outputs[0].cpu())

            val_img_lst.append(val_images.squeeze_()[0].cpu())
            val_msk_lst.append(val_outputs.squeeze_().cpu())
            val_tru_lst.append(val_labels.squeeze_().cpu())


            # print(val_images.shape)
            # print(val_labels.shape)
            # print(val_outputs.shape)

            # wandb.log({"masks/Image": [wandb.Image(val_images.squeeze_(), caption="Images")]})
            # wandb.log({"masks/true": [wandb.Image(val_labels.squeeze_(), caption="Mask/True")]})
            # wandb.log({"masks/pred": [wandb.Image(val_outputs.squeeze_(), caption="Mask/Pred")]})

            # plt.imshow(val_images.squeeze_()[0].cpu())
            # plt.savefig("inf_plot/img_"+args.data_view+"/"+val_images_name+".png", bbox_inches='tight')
            # plt.close()
            # plt.imshow(val_outputs.squeeze_().cpu())
            # plt.savefig("inf_plot/mask_"+args.data_view+"/"+val_labels_name+".png", bbox_inches='tight')
            # plt.close()
            # plt.imshow(val_labels.squeeze_().cpu())
            # plt.savefig("inf_plot/true_"+args.data_view+"/"+val_labels_name+".png", bbox_inches='tight')
            # plt.close()

            # f, axarr = plt.subplots(1,3)
            # axarr[0].imshow(val_images.squeeze_()[0].cpu())
            # axarr[0].set_title(args.data_view)
            # axarr[1].imshow(val_labels.squeeze_().cpu())
            # axarr[1].set_title("True")
            # axarr[2].imshow(val_outputs.squeeze_().cpu(), label="Predict")
            # axarr[2].set_title("Predict")
            # plt.savefig("inf_plot/compare/"+args.data_view+"_"+str(val_images_name)+".png")
            # plt.close()

            # ---------------------------------

        print("!!!!!! Minimum time: ", min(one_time))
        print("!!!!!! Average time: ", (sum(one_time[1:])-max(one_time[1:]))/(len(one_time)-2))

        # print_metrics(metrics, epoch_samples, 'val')

        val_img_mat = np.dstack(val_img_lst)
        val_msk_mat = np.dstack(val_msk_lst)
        val_tru_mat = np.dstack(val_tru_lst)

        val_img_mat_bk = val_trans_bk(val_img_mat.transpose(2,0,1))
        val_msk_mat_bk = val_trans_bk(val_msk_mat.transpose(2,0,1))
        val_tru_mat_bk = val_trans_bk(val_tru_mat.transpose(2,0,1))

        val_msk_d["data"] = val_msk_mat_bk
        val_tru_d["data"] = val_tru_mat_bk
        np.save("inf_plot/a_npz/"+args.data_view+"_pred_mat_d.npy", val_msk_d) 
        # np.save("inf_plot/a_npz/"+args.data_view+"_"+args.prun_config_file+"_pred_mat_d.npy", val_msk_d) 
        # np.save("inf_plot/a_npz/"+args.data_view+"_true_mat_d.npy", val_tru_d) 

        # print(val_img_mat_bk.shape)
        # print(val_msk_mat_bk.shape)
        # print(val_tru_mat_bk.shape)

        # np.savez("inf_plot/a_npz_full/"+args.data_view+"_img_mat.npz", val_img_mat)
        # np.savez("inf_plot/a_npz_full/"+args.data_view+"_pred_mat.npz", val_msk_mat)
        # np.savez("inf_plot/a_npz_full/"+args.data_view+"_true_mat.npz", val_tru_mat)

        # val_img_mat = nib.Nifti1Image(val_img_mat, np.eye(4))
        # val_msk_mat = nib.Nifti1Image(val_msk_mat, np.eye(4)) 
        # val_tru_mat = nib.Nifti1Image(val_tru_mat, np.eye(4)) 

        # nib.save(val_img_mat, "inf_plot/a_nii/"+args.data_view+"_img_mat.nii.gz")
        # nib.save(val_msk_mat, "inf_plot/a_nii/"+args.data_view+"_pred_mat.nii.gz")
        # nib.save(val_tru_mat, "inf_plot/a_nii/"+args.data_view+"_true_mat.nii.gz")

        # -----
        # np.savez("inf_plot/a_npz/"+args.data_view+"_img_mat.npz", val_img_mat_bk)
        # np.savez("inf_plot/a_npz/"+args.data_view+"_pred_mat.npz", val_msk_mat_bk)
        # np.savez("inf_plot/a_npz/"+args.data_view+"_true_mat.npz", val_tru_mat_bk)

        # val_img_mat_bk = nib.Nifti1Image(val_img_mat_bk, np.eye(4))
        # val_msk_mat_bk = nib.Nifti1Image(val_msk_mat_bk, np.eye(4)) 
        # val_tru_mat_bk = nib.Nifti1Image(val_tru_mat_bk, np.eye(4)) 

        # nib.save(val_img_mat_bk, "inf_plot/a_nii/"+args.data_view+"_img_mat_bk.nii.gz")
        # nib.save(val_msk_mat_bk, "inf_plot/a_nii/"+args.data_view+"_pred_ma_bkt.nii.gz")
        # nib.save(val_tru_mat_bk, "inf_plot/a_nii/"+args.data_view+"_true_ma_bkt.nii.gz")
        # -----

##########################################################################################################
def main(args):

    ###################################
    #         Path
    ###################################

    config = wandb.config
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_folder = './data/test_3'
    image_folder = 'img_'+args.data_view
    mask_folder = 'mask_'+args.data_view

    set_determinism(seed=0)

    ###################################
    #         Dataset
    ###################################

    images = sorted(glob(pjoin(data_folder, image_folder, '*.npz')), key = lambda e: int(e.split("/")[-1].split(".")[0]))
    segs = sorted(glob(pjoin(data_folder, mask_folder, '*.npz')), key = lambda e: int(e.split("/")[-1].split(".")[0]))

    # --------------
    data_img = np.load(images[0])['arr_0']
    data_shape = data_img.shape
    # --------------

    data_dicts = [
        {"img": image_name, "seg": label_name}
        for image_name, label_name in zip(images, segs)
    ]

    val_idx = int(len(images))

    val_files = data_dicts[-val_idx:]
    val_trans, val_trans_bk = get_transforms(args, data_shape[1], data_shape[2])

    val_ds = PersistentDataset(data=val_files, transform=val_trans, cache_dir=None)
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size, num_workers=1)

    ###################################
    #         Model
    ###################################
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("|||||||", device)

    if args.model == 'ResNetUNet':
        model = ResNetUNet(n_class=args.num_classes)
    else:
        print("Wrong model!")

    model = model.to(device) 

    # # ----------------------------
    # from torchsummary import summary
    # from prettytable import PrettyTable

    # summary(model, input_size=(3, 256, 256))

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
    
    ### Pruned Model
    # model_name = "ckpt_pruned_v1/retrain/{}_retrain_{}_{}{}.pt".format(args.data_view, args.prun_config_file, args.sparsity_type, args.ext)
    ### Original Model
    model_name = "ckpt/cvt_model/{}_ResNetUNet_best.pt".format(args.data_view)
    print(model_name)
    model.load_state_dict(torch.load(model_name))
    
    # ########## Convert tp onnx and Plot Network Structure ##############
    # import torch.onnx
    # torch.onnx.export(model, torch.zeros((1, 3, 256, 256)).cuda(), "ResUNet.onnx", opset_version=11)
    # os.system(netron ResUNet.onnx)
    # ####################################################################

    s1 = time.time()
    validation(args, model, device, val_loader, val_trans_bk)
    s2 = time.time()
    print("================= Total "+str(len(val_loader))+" imgs process: %f ms" % ((s2-s1)*1000))

    # print("+++++++++++++++++++++++++++++")
    # from ptflops import get_model_complexity_info
    # ### modified flops_counter.py for zero weight [conv_flops_counter_hook() & linear_flops_counter_hook()]
    # ### https://stackoverflow.com/questions/64551002/how-can-i-calculate-flops-and-params-without-0-weights-neurons-affected
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=False)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print("+++++++++++++++++++++++++++++")



            
        
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dims", default=256, type=list, 
                        help="Original shape: [(512, 512) -> (512, 512), (221, 512) -> (224, 512)]")
    parser.add_argument("--model", default='ResNetUNet', type=str,
                        help='Choose from [ResNetUNet, UNet, UNet11, UNet16, AlbuNet, LinkNet34]')
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument('--data_view', type=str, default="updown", metavar='N',
                        help="Choose from ['updown', 'leftright', 'frontback']")
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument('--sparsity_type', type=str, default='irregular', 
                        help="define sparsity_type: [irregular,column,filter]")
    parser.add_argument('--prun_config_file', type=str, default='config_ResNetUNet_0.9', 
                        help="prune config file")
    parser.add_argument('--ext', type=str, default="", metavar='N', help='extension for saved file')
    args = parser.parse_args()
    print(args)
    
    ###################################

    args, unknown = parser.parse_known_args()
    wandb.init(config=args)
    wandb.config.update(args)
    
    main(args)
