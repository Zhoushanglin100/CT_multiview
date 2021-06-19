#######################################################################
# Preprocessing Script for generating four label data                 #
# Fit for train_multi.py                                              #
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

def reverse_transform(inp):
    # inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

def do_img(im, i, idx, name, folder):

    os.makedirs(folder, exist_ok=True)

    im = np.fliplr(np.rot90(im, 1))                                         # [640, 480] -> [480, 640]
    im = np.reshape(im, (1, im.shape[0], im.shape[1])).astype(np.float32)   # [480, 640] -> [1, 480, 640]
    im = im.repeat(3, axis=0).astype(np.uint8)                              # [1, 480, 640] -> [3, 480, 640]
    
    plt.imshow(im.transpose([1,2,0]).squeeze())
    plt.savefig("dsply/img_"+str(i)+"_"+str(idx)+".png")
    
    zi = str(i).zfill(5)
    s_name = pjoin(folder, name + f'_{zi}')
    # np.savez(s_name+'.npz', im)

def do_msk(msk, i, idx, name, folder):

    os.makedirs(folder, exist_ok=True)

    msk = np.fliplr(np.rot90(msk, 1))
    msk_1 = np.reshape(msk, (1, msk.shape[0], msk.shape[1])).astype(np.float32)
    msk = np.concatenate([msk_1==i for i in range(1,8)], axis=0).astype(np.uint8)    # not include background

    # a = msk.squeeze().transpose((1, 2, 0))
    # a = np.clip(a, 0, 1)
    # a = (a * 255).astype(np.uint8)
    # plt.imshow(a)
    # plt.savefig("dsply/msk_"+str(i)+"_"+str(idx)+".png")

    a = msk_1.transpose([1,2,0]).squeeze(axis=2)
    plt.imshow(a)
    plt.savefig("dsply/msk_"+str(i)+"_"+str(idx)+".png")

    zi = str(i).zfill(5)
    s_name = pjoin(folder, name + f'_{zi}')
    # np.savez(s_name+'.npz', msk)

print("Generating Images!!")
for nii_idx, img_path in enumerate(nii_imgs):
    print(nii_idx, img_path)
    name = os.path.basename(img_path)[:7]

    img = nib.load(img_path).get_fdata()
    mask_path = [filename for filename in os.listdir(nii_folder) if filename.startswith(name+'_label')]
    mask_path = pjoin(nii_folder, mask_path[0])
    mask = nib.load(mask_path).get_fdata()

    img_updown = np.transpose(img, (2, 0, 1))
    mask_updown = np.transpose(mask, (2, 0, 1))

    img_leftright = np.transpose(img, (0, 1, 2))
    mask_leftright = np.transpose(mask, (0, 1, 2))

    img_frontback = np.transpose(img, (1, 0, 2))
    mask_frontback = np.transpose(mask, (1, 0, 2))

    mask_name_list = ["mask_updown", "mask_leftright", "mask_frontback"]
    img_name_list = ["img_updown", "img_leftright", "img_frontback"]

    # mask_type = [mask_updown, mask_leftright, mask_frontback]

    mask_type = [img_updown]
    for num in range(3):
        for idx, msk in enumerate(mask_type[num]):
            a = sum(sum(msk))
            if a != 0:
                do_msk(msk, nii_idx, idx, name, mask_name_list[num])
                do_img(img[idx], nii_idx, idx, name, img_name_list[num])
            print("Done ", str(idx))
        print("Done one type")
print('Done all!')