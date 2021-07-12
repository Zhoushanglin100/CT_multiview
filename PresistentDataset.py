from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset

class PersistentDataset(_TorchDataset):

    def __init__(self, data: Sequence, transform: Optional[Callable] = None, cache_dir = None) -> None:

        self.img = [np.load(e['img'])['arr_0'] for e in data] # input list of dict{'img':path, 'seg':path}; output list of np_array
        self.seg = [np.load(e['seg'])['arr_0'] for e in data] # input list of dict{'img':path, 'seg':path}; output list of np_array
                
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_img = self.img[index]
        data_seg = self.seg[index]
        
        data_img = data_img.astype(data_img.dtype)
        mina = np.min(data_img)
        maxa = np.max(data_img)
        minv = 0
        maxv = 1       
        norm = (data_img - mina) / (maxa - mina)
        data_img = np.asarray((norm*(maxv-minv))+minv)

        # print("===================")
        # print(torch.tensor(data_img).shape)
        # print(torch.tensor(data_seg).shape)
        # print("===================")

        return {"img": self.transform(data_img) if self.transform is not None else torch.tensor(data_img),
                "seg": self.transform(data_seg) if self.transform is not None else torch.tensor(data_seg)}


    def __getitem__(self, index: Union[int, slice, Sequence[int]]):

        return self._transform(index)


