import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.loader import DataLoader as pyg_DataLoader
from .augmentation import data_augmenter
from .hippocampus import *
from .fish import *
from .mutag import *
from .dogvscat import *


class Data_Loader:
    def __init__(self, args, split):
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
            args.crop,
            args.image_size
        )

        if args.dataset in hippocampus.__all__:
            self.data = hippocampus.__dict__[args.dataset](
                args.data_root,
                args.loading_method,
                args.no_label_data_portion,
                args.normalize,
                args.maxv,
                split,
                self.augmenter,
            )
        elif args.dataset in fish.__all__:
            self.data = fish.__dict__[args.dataset](
                args.data_root,
                args.loading_method,
                args.normalize,
                args.maxv,
                split,
                self.augmenter,                           
            )
        elif args.dataset in mutag.__all__:
            self.data = mutag.__dict__[args.dataset](
                args.data_root,
                args.fold,
                split,
            )
        elif args.dataset in dogvscat.__all__:
            self.data = dogvscat.__dict__[args.dataset](
                args.data_root,
                args.loading_method,
                args.normalize,
                args.maxv,
                split,
                self.augmenter if split in ["train"] else data_augmenter(args.data_format, "sham", 1., args.image_size),                           
            )

    def get_loader(self):
        indices = np.arange(len(self.data))
        sampler = SubsetRandomSampler(indices)

        if self.data_format in ['graph']:
            return  pyg_DataLoader(
                self.data,
                batch_size=self.batch_size,
                shuffle=True,
            )
        else:
            return DataLoader(
                self.data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=sampler,
                drop_last=True
            )
