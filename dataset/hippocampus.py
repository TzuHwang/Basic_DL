import os
from utils.file_dealer import json_loader, img_loader
from utils.misc import roll_a_dice

import torch
import torch.utils.data as data

__all__ = ["HippocampusSeg"]

class HippocampusSeg(data.Dataset):
    def __init__(self, data_root, loading_method, no_label_data_portion, normalize, maxv, split, augmenter):
        self.data_root = data_root
        self.split = split

        if self.split in ["train","val","train_final"]:
            self.input_root = f"{data_root}/original/train"
            self.anno_root = f"{data_root}/anno/train"
        else:
            self.input_root = f"{data_root}/original/test"
            self.anno_root = f"{data_root}/anno/test"

        self.valid_ids = self._split_train_val()
        self.no_labeled_pths = json_loader(f"{data_root}/no_label.json")

        self.normalize = normalize
        self.maxv = maxv
        self.loading_method = loading_method
        self.augmenter = augmenter
        self.no_label_data_portion = no_label_data_portion
        self.orig_pths, self.anno_pths = self._get_pths()

    def _get_pths(self):
        orig_pths, anno_pths = [], []
        if self.loading_method == "img_2D":
            ids = os.listdir(self.input_root)
            for id in ids:
                if len(self.valid_ids) != 0 and id not in self.valid_ids:
                    continue
                files = os.listdir(f"{self.input_root}/{id}")
                for file in files:
                    orig_pth = f"{self.input_root}/{id}/{file}"
                    anno_pth = f"{self.anno_root}/{id}/{file.split('_')[-1].replace('jpg','png')}"
                    if anno_pth in self.no_labeled_pths and roll_a_dice() <= self.no_label_data_portion:
                        continue
                    orig_pths.append(orig_pth)
                    anno_pths.append(anno_pth)
            return orig_pths, anno_pths

    def _split_train_val(self):
        js_dict = json_loader(f"{self.data_root}/train_val.json")
        train_ids = js_dict["training_set"]
        val_ids = js_dict["validation_set"]
        if self.split == "train":
            return train_ids
        elif self.split == "val":
            return val_ids
        else:
            return []
    
    def __len__(self):
        return len(self.orig_pths)

    def __getitem__(self, index):
        orig_pth, anno_pth = self.orig_pths[index], self.anno_pths[index]
        orig, anno = img_loader(orig_pth), img_loader(anno_pth, keep_color = True)
        aug_data = self.augmenter(image=orig, mask=anno)
        if self.normalize is True and self.maxv is not None:
            orig, anno = orig/self.maxv, anno/self.maxv
        return aug_data['image'], torch.moveaxis(aug_data['mask'], -1, 0)
    