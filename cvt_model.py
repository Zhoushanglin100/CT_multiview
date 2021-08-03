#############################################################################
# Script for convert torch later version model to formal/or general version #
#############################################################################

import os
import torch

ROOTDIR = "ckpt_pruned/retrain"

for subdir, dirs, files in os.walk(ROOTDIR):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".pt"):
            print (filepath)
            model = torch.load(filepath)
            torch.save(model, 
                        "ckpt_pruned/cvt_model/"+file, 
                        _use_new_zipfile_serialization=False)
