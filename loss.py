import torch
from torch import nn
from torch.nn import functional as F
# import utils
import numpy as np

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return dice.mean(), loss.mean()


# class LossMulti:
#     def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
#         if class_weights is not None:
#             nll_weight = utils.cuda(
#                 torch.from_numpy(class_weights.astype(np.float32)))
#         else:
#             nll_weight = None
#         self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
#         self.jaccard_weight = jaccard_weight
#         self.num_classes = num_classes

#     def __call__(self, outputs, targets):
#         loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

#         if self.jaccard_weight:
#             eps = 1e-15
#             for cls in range(self.num_classes):
#                 jaccard_target = (targets == cls).float()
#                 jaccard_output = outputs[:, cls].exp()
#                 intersection = (jaccard_output * jaccard_target).sum()

#                 union = jaccard_output.sum() + jaccard_target.sum()
#                 loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
#         return loss