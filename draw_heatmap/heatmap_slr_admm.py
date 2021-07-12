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



# minimum=0
# maximum=0.02

# config_file = 'profile/config_heatmap.yaml' #just leave 5 layers with the highest sparsity and delete other in the config file
# model_path = "tusimple_18.pth" #pretrained model

# model = parsingNet(pretrained = True, backbone='18',cls_dim = (101, 56, 4),use_aux=False).cuda()

# state_dict = torch.load(model_path, map_location = 'cpu')['model']
# compatible_state_dict = {}
# for k, v in state_dict.items():
#     if 'module.' in k:
#         compatible_state_dict[k[7:]] = v
#     else:
#         compatible_state_dict[k] = v
# model.load_state_dict(compatible_state_dict, strict = False)


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



# fig, axs = plt.subplots(1,3)

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
#     im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, 
#                                     ax=axs[0], cmap="gnuplot", cbarlabel=name, title="Before Pruning") # cbarlabel=name
#     # im, cbar = myheatmap.heatmap(minimum, # maximum, harvest, vegetables, farmers, 
#     #                                 ax=axs[j,0], cmap="gnuplot", cbarlabel=name, title="Before Pruning") # cbarlabel=name
#     j=j+1


# model_path = "loggger_v2/model_prunned/admm_train/test/admmtest_slr_0.99.pth" #model after slr training (not hardpruned)
# model = parsingNet(pretrained = True, backbone='18',cls_dim = (101, 56, 4),use_aux=False).cuda()

# state_dict = torch.load(model_path, map_location = 'cpu')['model']
# compatible_state_dict = {}
# for k, v in state_dict.items():
#     if 'module.' in k:
#         compatible_state_dict[k[7:]] = v
#     else:
#         compatible_state_dict[k] = v
# model.load_state_dict(compatible_state_dict, strict = False)



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
#     im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, 
#                                     ax=axs[1], cmap="gnuplot", 
#                                     title="After SLR Training") # cbarlabel=name
#     # im, cbar = myheatmap.heatmap(minimum, # maximum, harvest, vegetables, farmers, 
#     #                             ax=axs[j,1], cmap="gnuplot", 
#     #                             title="After SLR Training") # cbarlabel=name
#     j=j+1
    

# model_path = "loggger_v2/model_prunned/mask_retrain/test/masktest_slr_0.99_10.pth" #slr model after retraining
# model = parsingNet(pretrained = True, backbone='18',cls_dim = (101, 56, 4),use_aux=False).cuda()

# state_dict = torch.load(model_path, map_location = 'cpu')['model']
# compatible_state_dict = {}
# for k, v in state_dict.items():
#     if 'module.' in k:
#         compatible_state_dict[k[7:]] = v
#     else:
#         compatible_state_dict[k] = v
# model.load_state_dict(compatible_state_dict, strict = False)


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
#     im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, 
#                                     ax=axs[2], cmap="gnuplot", title="After Hardpruning") # cbarlabel=name
#     # im, cbar = myheatmap.heatmap(minimum, # maximum, harvest, vegetables, farmers, ax=axs[j,2], 
#     #                         cmap="gnuplot", title="After Hardpruning") # cbarlabel=name
#     j=j+1



# #fig.tight_layout()
# fig.set_size_inches(10,10)
# fig.tight_layout(pad=0.6)

# #plt.show()
# fig.savefig('heatmap_slr_0.99.pdf')


#################################################################

minimum=0.00002
maximum=0.00005

# minimum=0
# maximum=0.01


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



fig, axs = plt.subplots(1,3)

j=0
for name, weight in model.named_parameters():

    if name not in prune_ratios:
        continue
    print("name: ", name, "weight.shape = ", weight.shape)  

    conv = np.abs(weight.reshape(weight.shape[0],-1).cpu().detach().numpy())
    conv = weight.reshape(weight.shape[0], -1)
    conv = conv[:, 1000:1800]
    conv = conv[0:250, 350:450]   
    print("!!! Before", conv.shape)
    array = np.array(conv.cpu().detach().numpy())
    harvest=abs(array)

    vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, 
                                    ax=axs[0], cmap="gnuplot", cbarlabel=name, 
                                    title="Before Pruning") # cbarlabel=name
    # im, cbar = myheatmap.heatmap(minimum, # maximum, harvest, vegetables, farmers, 
    #                                 ax=axs[j,0], cmap="gnuplot", cbarlabel=name, 
    #                                 title="Before Pruning") # cbarlabel=name
    j=j+1


model_path = "loggger_v2/model_prunned/admm_train/test/admmtest_slr_0.99.pth" #model after slr training (not hardpruned)
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
    conv = conv[:, 1000:1800]
    conv = conv[0:250:, 350:450]    
    print("!!! slr", conv.shape)
    array = np.array(conv.cpu().detach().numpy())
    harvest=abs(array)

    vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    im, cbar = myheatmap.heatmap(minimum, maximum, harvest, vegetables, farmers, 
                                    ax=axs[1], cmap="gnuplot", title="After SLR Training") # cbarlabel=name
    # im, cbar = myheatmap.heatmap(minimum, # maximum, harvest, vegetables, farmers, 
    #                                 ax=axs[j,1], cmap="gnuplot", title="After SLR Training") # cbarlabel=name                                    
    j=j+1


model_path = "loggger_v2/model_prunned/admm_train/test/admmtest_admm_0.99.pth" #model after admm training (not hardpruned)
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
    conv = conv[:, 1000:1800]
    conv = conv[0:250:, 350:450]
    print("!!! admm", conv.shape)
    array = np.array(conv.cpu().detach().numpy())
    harvest=abs(array)

    vegetables = None # ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    farmers = None #["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    im, cbar = myheatmap.heatmap(minimum, maximum,harvest, vegetables, farmers, 
                                    ax=axs[2], cmap="gnuplot", title="After ADMM Training") # cbarlabel=name
    # im, cbar = myheatmap.heatmap(minimum, # maximum,harvest, vegetables, farmers, 
    #                                 ax=axs[j,2], cmap="gnuplot", title="After ADMM Training") # cbarlabel=name
    j=j+1



#fig.tight_layout()
fig.set_size_inches(15, 4)
fig.tight_layout(pad=0.6)

#plt.show()
fig.savefig('heatmap_admm_0.99_0.99.pdf')