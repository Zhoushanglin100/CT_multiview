import myheatmap
import yaml
# from resnet import resnet50
import os
import time
import torch

import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from model.model import parsingNet





minimum=0
maximum=0.02

# model = resnet50()
config_file = 'profile/config_heatmap.yaml' #just leave 5 layers with the highest sparsity and delete other in the config file
model_path = "tusimple_18.pth" #pretrained model

model = parsingNet(pretrained = True, backbone='18',cls_dim = (101, 56, 4),use_aux=False).cuda()

state_dict = torch.load(model_path, map_location = 'cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v
model.load_state_dict(compatible_state_dict, strict = False)



# Different block size for different layers:
with open(config_file, "r") as stream:
    try:
        raw_dict = yaml.load(stream)
        prune_ratios = raw_dict['prune_ratios']
        print(prune_ratios)
    except yaml.YAMLError as exc:
        print(exc)



i=0
for name, weight in model.named_parameters():
    if name not in prune_ratios:
        continue
    i=i+1



fig, axs = plt.subplots(i,3)

j=0
for name, weight in model.named_parameters():


    if name not in prune_ratios:
        continue
    print("name: ", name, "weight.shape = ", weight.shape)  


    conv = np.abs(weight.reshape(weight.shape[0],-1).cpu().detach().numpy())
    conv = weight.reshape(weight.shape[0], -1)
    array = np.array(conv.cpu().detach().numpy())
    harvest=abs(array)

    vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, ax=axs[j,0], cmap="gnuplot", 
                            cbarlabel=name, title="Before Pruning") # cbarlabel=name
    j=j+1


model_path = "/home/shz15022/obj_det/wp_model/LN/admmtest_admm_nobn_0.7.pth" #model after slr training (not hardpruned)
model = parsingNet(pretrained = True, backbone='18',cls_dim = (101, 56, 4),use_aux=False).cuda()

state_dict = torch.load(model_path, map_location = 'cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v
model.load_state_dict(compatible_state_dict, strict = False)



j=0
for name, weight in model.named_parameters():


    if name not in prune_ratios:
        continue
    print("name: ", name, "weight.shape = ", weight.shape)  

    conv = np.abs(weight.reshape(weight.shape[0],-1).cpu().detach().numpy())
    conv = weight.reshape(weight.shape[0], -1)
    array = np.array(conv.cpu().detach().numpy())
    harvest=abs(array)

    vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, ax=axs[j,1], cmap="gnuplot", 
                            title="After ADMM Training") # cbarlabel=name
    j=j+1
    

model_path = "/home/shz15022/obj_det/wp_model/LN/masktest_admm_nobn_0.7_10.pth" #slr model after retraining
model = parsingNet(pretrained = True, backbone='18',cls_dim = (101, 56, 4),use_aux=False).cuda()

state_dict = torch.load(model_path, map_location = 'cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v
model.load_state_dict(compatible_state_dict, strict = False)


j=0
for name, weight in model.named_parameters():


    if name not in prune_ratios:
        continue
    print("name: ", name, "weight.shape = ", weight.shape)  

    conv = np.abs(weight.reshape(weight.shape[0],-1).cpu().detach().numpy())
    conv = weight.reshape(weight.shape[0], -1)
    array = np.array(conv.cpu().detach().numpy())
    harvest=abs(array)

    vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, ax=axs[j,2], 
                            cmap="gnuplot", title="After Retrain") # cbarlabel=name
    j=j+1



#fig.tight_layout()
fig.set_size_inches(15,15)
fig.tight_layout(pad=0.6)

#plt.show()
fig.savefig('heatmap_lane.pdf')


# #################################################################

# minimum=0
# maximum=0.02

# model = resnet50()
# config_file = 'profile/config_res50.yaml'  #just leave 5 layers with the highest sparsity and delete other in the config file
# model_path = "model/resnet50.pt" #pretrained model


# #state_dict = state_dict.get('model', state_dict)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# # Different block size for different layers:
# with open(config_file, "r") as stream:
#     try:
#         raw_dict = yaml.load(stream)
#         prune_ratios = raw_dict['prune_ratios']
#         print(prune_ratios)
#     except yaml.YAMLError as exc:
#         print(exc)



# i=0
# for name, weight in model.named_parameters():
#     if name not in prune_ratios:
#         continue
#     i=i+1



# fig, axs = plt.subplots(i,3)

# j=0
# for name, weight in model.named_parameters():


#     if name not in prune_ratios:
#         continue
#     print("name: ", name, "weight.shape = ", weight.shape)  


#     conv = np.abs(weight.reshape(weight.shape[0],-1).cpu().detach().numpy())
#     conv = weight.reshape(weight.shape[0], -1)
#     array = np.array(conv.cpu().detach().numpy())
#     harvest=abs(array)

#     vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
#     farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
#     im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, ax=axs[j,0], cmap="gnuplot", cbarlabel=name, title="Before Pruning") # cbarlabel=name
#     j=j+1

# model = resnet50()
# model_path = "mnist_0.1_config_res50_irregular_slr.pt" #model after slr training (not hardpruned)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# j=0
# for name, weight in model.named_parameters():


#     if name not in prune_ratios:
#         continue
#     print("name: ", name, "weight.shape = ", weight.shape)  

#     conv = np.abs(weight.reshape(weight.shape[0],-1).cpu().detach().numpy())
#     conv = weight.reshape(weight.shape[0], -1)
#     array = np.array(conv.cpu().detach().numpy())
#     harvest=abs(array)

#     vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
#     farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
#     im, cbar = myheatmap.heatmap(minimum, maximum,harvest, vegetables, farmers, ax=axs[j,1], cmap="gnuplot", title="After SLR Training") # cbarlabel=name
#     j=j+1
# model = resnet50()
# model_path = "mnist_0.1_config_res50_irregular_admm.pt" #model after admm training (not hardpruned)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# j=0
# for name, weight in model.named_parameters():


#     if name not in prune_ratios:
#         continue
#     print("name: ", name, "weight.shape = ", weight.shape)  

#     conv = np.abs(weight.reshape(weight.shape[0],-1).cpu().detach().numpy())
#     conv = weight.reshape(weight.shape[0], -1)
#     array = np.array(conv.cpu().detach().numpy())
#     harvest=abs(array)

#     vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
#     farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
#     im, cbar = myheatmap.heatmap(minimum, maximum,harvest, vegetables, farmers, ax=axs[j,2], cmap="gnuplot", title="After ADMM Training") # cbarlabel=name
#     j=j+1



# #fig.tight_layout()
# fig.set_size_inches(10,10)
# fig.tight_layout(pad=0.6)

# #plt.show()
# fig.savefig('heatmap_resnet50_compare.pdf')