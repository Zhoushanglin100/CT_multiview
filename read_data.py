import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import sys

from os import listdir
from os.path import isfile, join

import nibabel as nib

###########################################

# fldr_name = ['updown', 'leftright', 'frontback']

# for fldr in fldr_name:
#     lst = [f for f in listdir("data/3d/img_"+fldr) if isfile(join("data/3d/img_"+fldr, f))]

#     for f in lst:
#         data_img = np.load("data/3d/img_"+fldr+"/"+f)['arr_0']
        
#         ext_name = f[3:]
#         data_msk = np.load("data/3d/mask_"+fldr+"/msk"+ext_name)['arr_0']
        
#         print(data_img.shape, data_msk.shape)

#         # if (data_msk.shape != (512, 512)):
#         #     print("msk"+ext_name)
#         #     print(data_msk.shape)

#         # if (data_img.shape != (3, 512, 512)):
#         #     print(f)
#         #     print(data_img.shape)


# for i in range(89, 400):
    
    # img = "data/3d/img_updown/img_0_"+str(i)+".npz"
    # msk = "data/3d/mask_updown/msk_0_"+str(i)+".npz"

    # data_img = np.load(img)['arr_0']
    # data_msk = np.load(msk)['arr_0']

    # print(data_img.shape, data_msk.shape)

# plt.imshow(data_img)
# plt.savefig("img_rev.png")

# plt.imshow(data_msk)
# plt.savefig("msk_rev.png")

# print("aaa")



##############################################################
# to run this part:
# python3 read_data.py arg1
#   @arg1: prune rate, [0.5, 0.9, 0.99]
##############################################################

nii_imgs = ["/data/medical/ImageCHD_dataset/ct_1178_image.nii.gz"]
nii_masks = ["/data/medical/ImageCHD_dataset/ct_1178_label.nii.gz"]

img = nib.load(nii_imgs[0]).get_fdata()
msk = nib.load(nii_masks[0]).get_fdata()

## convert to 2 classes (blood pool vs background)
msk[np.where(msk == 5)] = 0
msk[np.where(msk != 0)] = 1

# np.savez("inf_plot/a_npz_full/msk_ResNetUNet.npz", msk)

# ### Save back to nii for same affine
# img_full_nii = nib.Nifti1Image(img, np.eye(4)) 
# msk_full_nii = nib.Nifti1Image(msk, np.eye(4))
# nib.save(img_full_nii, "inf_plot/a_nii/img_all.nii.gz")
# nib.save(msk_full_nii, "inf_plot/a_nii/msk_all.nii.gz")

img_shp = img.shape

### Deal with pred
file_name = ['updown', 'leftright', 'frontback']

for view in file_name:

    print("------>", view)

    # pred_name = "inf_plot/a_npz/"+view+"_config_ResNetUNet_"+sys.argv[1]+"_pred_mat_d.npy"
    pred_name = "inf_plot/a_npz/"+view+"_pred_mat_d.npy"
    # true_name = "inf_plot/a_npz/"+view+"_true_mat_d.npy"

    read_pred = np.load(pred_name, allow_pickle='TRUE').item()
    # read_true = np.load(true_name, allow_pickle='TRUE').item()

    pred = read_pred["data"]
    pred_first = int(read_pred["first"])
    pred_last = int(read_pred["last"])

    pred_bk = np.rot90(np.flip(pred, axis=2), 3, axes=(1,2))
    # pred_bk = np.flip(np.rot90(pred, 1, axes=(1,2)), axis=1)

    if view == "updown":
        pred_bk = pred_bk.transpose(1,2,0)
        
        pred_shp = pred_bk.shape
        pred_app_1 = np.zeros([pred_shp[0], pred_shp[1], pred_first])
        pred_app_2 = np.zeros([pred_shp[0], pred_shp[1], img_shp[2]-pred_last-1])
        pred_full = np.concatenate((pred_app_1, pred_bk, pred_app_2), axis=-1)
        pred_ud = pred_full
        print("updown shape:", pred_ud.shape)

    elif view == "leftright":
        pred_bk = pred_bk.transpose(0,1,2)
        pred_shp = pred_bk.shape
        pred_app_1 = np.zeros([pred_first, pred_shp[1], pred_shp[2]])
        pred_app_2 = np.zeros([img_shp[0]-pred_last-1, pred_shp[1], pred_shp[2]])
        pred_full = np.concatenate((pred_app_1, pred_bk, pred_app_2), axis=0)
        pred_lf = pred_full
        print("leftright shape:", pred_lf.shape)

    elif view == "frontback":
        pred_bk = pred_bk.transpose(1,0,2)
        pred_shp = pred_bk.shape
        pred_app_1 = np.zeros([pred_shp[0], pred_first, pred_shp[2]])
        pred_app_2 = np.zeros([pred_shp[0], img_shp[1]-pred_last-1, pred_shp[2]])
        pred_full = np.concatenate((pred_app_1, pred_bk, pred_app_2), axis=1)
        pred_fb = pred_full
        print("frontback shape:", pred_fb.shape)

    print(view, float(sys.argv[1]), " save to npz!")
    np.savez("inf_plot/a_npz_full/"+view+"_ResNetUNet_"+sys.argv[1]+"_pred_full.npz", pred_full)

    # print(view, float(sys.argv[1]), " save to nii!")
    # pred_full_nii = nib.Nifti1Image(pred_full, np.eye(4)) 
    # nib.save(pred_full_nii, "inf_plot/a_nii/"+view+"_config_ResNetUNet_"+sys.argv[1]+"_pred_full.nii.gz")

pred_all = np.round((pred_ud+pred_lf+pred_fb)/3)
# pred_max = np.maximum.reduce([pred_ud,pred_lf,pred_fb])
# pred_min = np.minimum.reduce([pred_ud,pred_lf,pred_fb])

print(pred_all.shape)
# print(pred_max.shape)
# print(pred_min.shape)

np.savez("inf_plot/a_npz_full/pred_mj_"+sys.argv[1]+".npz", pred_all)


# pred_all_nii = nib.Nifti1Image(pred_all, np.eye(4))
# nib.save(pred_all_nii, "inf_plot/a_nii/"+sys.argv[1]+"_pred_all.nii.gz")

# pred_max_nii = nib.Nifti1Image(pred_max, np.eye(4))
# nib.save(pred_max_nii, "inf_plot/a_nii_1/pred_max.nii.gz")

# pred_min_nii = nib.Nifti1Image(pred_min, np.eye(4))
# nib.save(pred_min_nii, "inf_plot/a_nii_1/pred_min.nii.gz")

print("Done!!")