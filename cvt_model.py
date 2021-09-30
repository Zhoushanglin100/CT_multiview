#############################################################################
# Script for convert torch later version model to formal/or general version #
#############################################################################

import os
import torch

ROOTDIR = "ckpt/best_1"

for subdir, dirs, files in os.walk(ROOTDIR):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".pt"):
            print (filepath)
            model = torch.load(filepath, map_location ='cpu')
            torch.save(model, 
                        "ckpt/cvt_model/"+file, 
                        _use_new_zipfile_serialization=False)
