import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

###########################################
def reverse_transform(inp):
    inp = inp.transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

i = 95

img = "data/img_leftright/img_0_"+str(i)+".npz"
msk = "data/mask_leftright/msk_0_"+str(i)+".npz"

data_img = np.load(img)['arr_0']
data_msk = np.load(msk)['arr_0']

print(data_img.shape, data_msk.shape)

plt.imshow(data_img)
plt.savefig("img_rev.png")

plt.imshow(data_msk)
plt.savefig("msk_rev.png")

print("aaa")