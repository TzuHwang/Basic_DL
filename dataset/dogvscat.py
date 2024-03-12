import os, natsort
import numpy as np
from utils.file_dealer import json_loader, json_saver, img_loader
from utils.misc import roll_a_dice

import torch
import torch.utils.data as data

__all__ = ["DogCatCls"]

class DogCatCls(data.Dataset):
    def __init__(self, data_root, loading_method, normalize, maxv, split, augmenter):
        self.data_root = data_root

        _, self.species, _ = next(os.walk(self.data_root), ([],[],[]))
        self.split = split
        if self.split in ["train", "val"]:
            self.input_root = f"{data_root}/train"
            self.train_val_split = self._split_train_val()[self.split]
        else:
            self.input_root = f"{data_root}/test1"
            self.train_val_split = None


        self.normalize = normalize
        self.maxv = maxv

        self.augmenter = augmenter
        self.orig_pths, self.categories = self._get_pths()
    
    def _get_pths(self):
        orig_pths, categories = [], []
        files = natsort.natsorted(os.listdir(self.input_root))
        for file in files:
            if self.train_val_split is not None:
                if file not in self.train_val_split:
                    continue
            orig_pths.append(f"{self.input_root}/{file}")
            categories.append(1) if "dog" in file else categories.append(0)
        return orig_pths, categories
    
    def _split_train_val(self):
        if os.path.exists(f"{self.data_root}/train_val_split.json") is False:
            train_val_split = {
                "train":[],
                "val":[]
            }
            files = natsort.natsorted(os.listdir(self.input_root))
            for file in files:
                chance = roll_a_dice()
                train_val_split["train"].append(file) if chance<=0.8 else train_val_split["val"].append(file)
            json_saver(f"{self.data_root}/train_val_split.json", train_val_split)
        train_val_split = json_loader(f"{self.data_root}/train_val_split.json")
        return train_val_split
            
    def __len__(self):
        return len(self.orig_pths)

    def __getitem__(self, index):
        orig_pth, cat = self.orig_pths[index], self.categories[index]
        orig = img_loader(orig_pth, keep_color = True)
        aug_data = self.augmenter(image=orig)
        orig = aug_data['image']
        if self.normalize is True and self.maxv is not None:
            orig = orig/self.maxv
        return orig , cat