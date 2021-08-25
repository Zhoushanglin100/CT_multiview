##########################################
# Script to calculate 3D IOU (dice coef) #
# To run the script:                     #
# python3 ck3Dacc.py arg1                #
#   @arg1: prune rate, [0.5, 0.9, 0.99]  #
##########################################

import numpy as np
import sys

# ------------------------------------------

def dice_coef(pred, target, smooth = 1.):

    pred = pred.copy()
    target = target.copy()

    volume_sum = target.sum() + pred.sum()

    if volume_sum == 0:
        return np.NaN

    # volume_intersect = (target & pred).sum()
    volume_intersect = np.where((target==pred), target, 0).sum()

    dice = (2 * volume_intersect + smooth) / (volume_sum + smooth)
    
    return dice

# ------------------------------------------


msk_true = np.load("inf_plot/a_npz_full/msk_ResNetUNet.npz")["arr_0"]
pred_full = np.load("inf_plot/a_npz_full/pred_mj_"+sys.argv[1]+".npz")["arr_0"]

pred_ud = np.load("inf_plot/a_npz_full/updown_ResNetUNet_"+sys.argv[1]+"_pred_full.npz")["arr_0"]
pred_lr = np.load("inf_plot/a_npz_full/leftright_ResNetUNet_"+sys.argv[1]+"_pred_full.npz")["arr_0"]
pred_fb = np.load("inf_plot/a_npz_full/frontback_ResNetUNet_"+sys.argv[1]+"_pred_full.npz")["arr_0"]


iou_all = dice_coef(pred_full, msk_true)
iou_ud = dice_coef(pred_ud, msk_true)
iou_lr = dice_coef(pred_lr, msk_true)
iou_fb = dice_coef(pred_fb, msk_true)

print("Overall: ", iou_all)
print("Updown: ", iou_ud)
print("Leftright: ", iou_lr)
print("Frontback: ", iou_fb)