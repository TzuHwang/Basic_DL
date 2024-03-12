import os
from utils.file_dealer import json_loader, img_loader
from utils.misc import roll_a_dice

import torch
import torch.utils.data as data

__all__ = ["coco2017"]

class coco(data.Dataset):
    def __init__(self, data_root, loading_method, normalize, maxv, split, augmenter) -> None:
        self.data_root = data_root