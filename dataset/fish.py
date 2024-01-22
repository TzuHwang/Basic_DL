import os, natsort
import numpy as np
from utils.file_dealer import json_loader, img_loader
from utils.misc import roll_a_dice

import torch
import torch.utils.data as data

__all__ = ["FishSeg", "FishCls"]

'''
The original images of redseabream do not match their annotations in number 00381~00400 
'''
datafilter = {
    "RedSeaBream": list(range(381, 401)),
    "SeaBass": list(range(701, 801)),
}

def filter_out(cat, id):
    if cat in datafilter:
        idx = int(id.split(".")[0])
        return idx in datafilter[cat]
    else:
        return False

def to_one_hot(anno, category, channel_num):
    _, h, w = anno.shape
    one_hot_anno = torch.zeros(channel_num, h, w)
    one_hot_anno[0] = torch.ones(anno.shape) - anno
    one_hot_anno[category] = anno
    return one_hot_anno

class FishSeg(data.Dataset):
    def __init__(self, data_root, loading_method, normalize, maxv, mod, augmenter):
        self.data_root = data_root
        _, species, _ = next(os.walk(self.data_root), ([],[],[]))
        if loading_method == "all":
            self.species = species
        else:
            assert loading_method in species
            self.species = [loading_method]
        self.mod = mod
        self.normalize = normalize
        self.maxv = maxv

        self.augmenter = augmenter
        self.split_portion = [0.64, 0.8, 1.0]
        self.orig_pths, self.anno_pths, self.categories = self._get_pths()
    
    def _get_pths(self):
        orig_pths, anno_pths, categories = [], [], []
        for i, s in enumerate(self.species):           
            input_root = f"{self.data_root}/{s}/original"
            anno_root = f"{self.data_root}/{s}/anno" 

            files = natsort.natsorted(os.listdir(input_root))
            datasize = len(files)
            if self.mod == "train":
                start, end = 0, int(datasize*self.split_portion[0])
            elif self.mod == "val":
                start, end = int(datasize*self.split_portion[0]), int(datasize*self.split_portion[1])
            elif self.mod == "test":
                start, end = int(datasize*self.split_portion[1]), int(datasize*self.split_portion[2])
            files = files[start: end]
            for file in files:
                if filter_out(s, file):
                    continue
                orig_pths.append(f"{input_root}/{file}")
                anno_pths.append(f"{anno_root}/{file}")
                categories.append(i + 1)
        return orig_pths, anno_pths, categories

    def __len__(self):
        return len(self.orig_pths)

    def __getitem__(self, index):
        orig_pth, anno_pth, cat = self.orig_pths[index], self.anno_pths[index], self.categories[index]
        orig, anno = img_loader(orig_pth, keep_color = True), img_loader(anno_pth)
        aug_data = self.augmenter(image=orig, mask=anno)
        orig , anno = aug_data['image'], aug_data['mask'].unsqueeze(0)
        if self.normalize is True and self.maxv is not None:
            orig, anno = orig/self.maxv, anno/self.maxv
        if np.max(self.categories) != 1:
            channel_num = np.max(self.categories) + 1
            anno = to_one_hot(anno, cat, channel_num)
        return orig , anno
    
class FishCls(data.Dataset):
    def __init__(self, data_root, loading_method, normalize, maxv, mod, augmenter):
        self.data_root = data_root
        _, self.species, _ = next(os.walk(self.data_root), ([],[],[]))
        self.mod = mod
        self.normalize = normalize
        self.maxv = maxv

        self.augmenter = augmenter
        self.split_portion = [0.64, 0.8, 1.0]
        self.orig_pths, self.anno_pths, self.categories = self._get_pths()
    
    def _get_pths(self):
        orig_pths, anno_pths, categories = [], [], []
        for i, s in enumerate(self.species):
            input_root = f"{self.data_root}/{s}/original"
            anno_root = f"{self.data_root}/{s}/anno" 

            files = natsort.natsorted(os.listdir(input_root))
            datasize = len(files)
            if self.mod == "train":
                start, end = 0, int(datasize*self.split_portion[0])
            elif self.mod == "val":
                start, end = int(datasize*self.split_portion[0]), int(datasize*self.split_portion[1])
            elif self.mod == "test":
                start, end = int(datasize*self.split_portion[1]), int(datasize*self.split_portion[2])
            files = files[start: end]
            for file in files:
                orig_pths.append(f"{input_root}/{file}")
                anno_pths.append(f"{anno_root}/{file}")
                categories.append(i)
        return orig_pths, anno_pths, categories

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