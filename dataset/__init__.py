import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from .augmentation import data_augmenter
from .fish import *


class Data_Loader:
    def __init__(self, args, mod):
        self.data_format = args.data_format
        self.loading_method = args.loading_method
        self.no_label_data_portion = args.no_label_data_portion
        self.normalize = args.normalize
        self.maxv = args.maxv

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.augmenter = data_augmenter(
            args.data_format,
            args.aug,
            args.crop
        )

        if args.dataset in fish.__all__:
            self.data = fish.__dict__[args.dataset](
                args.data_root,
                args.loading_method,
                args.normalize,
                args.maxv,
                mod,
                self.augmenter,                           
            )
            pass

    def get_loader(self):
        indices = np.arange(len(self.data))
        sampler = SubsetRandomSampler(indices)

        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True)
