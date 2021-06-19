#######################################################################
# Preprocessing Script for generating label data                      #
# Only one label: blood pool vs background                            # 
# Fit for train_multi.py                                              #
# generated data saved to ./data                                      # 
#######################################################################

import nibabel as nib
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from varname import nameof

##################################################

pjoin = os.path.join

data_home_folder = './ImageCHD_dataset'

nii_folder = pjoin(data_home_folder)
img_folder = pjoin('data', 'images')
mask_folder = pjoin('data', 'masks')

# os.makedirs(img_folder, exist_ok=True)
# os.makedirs(mask_folder, exist_ok=True)

niis = glob(pjoin(nii_folder, '*.nii.gz'))
nii_imgs = sorted([n for n in niis if 'image' in n])
nii_masks = sorted([n for n in niis if 'label' in n])

##################################################

def reverse_transform(inp):
    # inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

def do_img(im, i, idx, name, folder):

    im = np.fliplr(np.rot90(im, 1))                                         # [640, 480] -> [480, 640]
    im_1 = im.astype(np.int8)
    im = np.reshape(im, (1, im.shape[0], im.shape[1])).astype(np.float32)   # [480, 640] -> [1, 480, 640]
    # im = im.repeat(3, axis=0).astype(np.uint8)                              # [1, 480, 640] -> [3, 480, 640]
    
    np.savez("data/"+folder+"/img_"+str(i)+"_"+str(idx)+'.npz', im_1)

    # plt.imshow(im.transpose([1,2,0]).squeeze())
    # plt.savefig("img.png")
    # plt.savefig("dsply/"+folder+"/img_"+str(i)+"_"+str(idx)+".png")
    
    # zi = str(i).zfill(5)
    # s_name = pjoin(folder, name + f'_{zi}')
    # np.savez("data", s_name+'.npz', im)


def do_msk(msk, i, idx, name, folder):

    msk = np.fliplr(np.rot90(msk, 1))
    msk_1 = msk.astype(np.int8)
    msk = np.reshape(msk, (1, msk.shape[0], msk.shape[1])).astype(np.float32)
    # msk = np.concatenate([msk_1==i for i in range(1,8)], axis=0).astype(np.uint8)    # not include background

    np.savez("data/"+folder+"/msk_"+str(i)+"_"+str(idx)+'.npz', msk_1)

    # a = msk.transpose([1,2,0]).squeeze(axis=2)
    # plt.imshow(a)
    # plt.savefig("dsply/"+folder+"/msk_"+str(i)+"_"+str(idx)+".png")

    # zi = str(i).zfill(5)
    # s_name = pjoin("data", folder, name + f'_{zi}')
    # np.savez(s_name+'.npz', msk)

##################################################

print("Generating Images!!")
for nii_idx, img_path in enumerate(nii_imgs):
# for nii_idx, img_path in enumerate(nii_imgs[:2]):

    print(nii_idx, img_path)
    name = os.path.basename(img_path)[:7]

    img = nib.load(img_path).get_fdata()
    mask_path = [filename for filename in os.listdir(nii_folder) if filename.startswith(name+'_label')]
    mask_path = pjoin(nii_folder, mask_path[0])
    mask = nib.load(mask_path).get_fdata()

    ## convert to 2 classes (blood pool vs background)
    mask[np.where(mask == 5)] = 0
    mask[np.where(mask != 0)] = 1
    ###

    img_updown = np.transpose(img, (2, 0, 1))
    mask_updown = np.transpose(mask, (2, 0, 1))

    img_leftright = np.transpose(img, (0, 1, 2))
    mask_leftright = np.transpose(mask, (0, 1, 2))

    img_frontback = np.transpose(img, (1, 0, 2))
    mask_frontback = np.transpose(mask, (1, 0, 2))

    mask_name_list = ["mask_updown", "mask_leftright", "mask_frontback"]
    img_name_list = ["img_updown", "img_leftright", "img_frontback"]

    # mask_type = [mask_updown, mask_leftright, mask_frontback]

    mask_type = [mask_updown, mask_leftright, mask_frontback]
    img_type = [img_updown, img_leftright, img_frontback]


    # for num in range(3):
    for num in [0]:

        # print("Generating ", mask_name_list[num])
        # os.makedirs("dsply/"+mask_name_list[num], exist_ok=True)
        # os.makedirs("dsply/"+img_name_list[num], exist_ok=True)
        os.makedirs("data/"+mask_name_list[num], exist_ok=True)
        os.makedirs("data/"+img_name_list[num], exist_ok=True)

        for idx, msk in enumerate(mask_type[num]):
            # if (idx > 100) and (idx < 110):
            img = img_type[num]
            a = sum(sum(msk))
            if a != 0:
                print(nii_idx, idx, name, mask_name_list[num])
                do_msk(msk, nii_idx, idx, name, mask_name_list[num])
                do_img(img[idx], nii_idx, idx, name, img_name_list[num])
            # print("Done ", str(idx))
        print("Done one type")
print('Done all!')