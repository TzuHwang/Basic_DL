import torch, os, natsort
import torch.nn.functional as F
import numpy as np

import torch_geometric.data as data
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.io import read_txt_array
from torch_geometric.io.tu import split
from torch_geometric.utils import coalesce, cumsum, remove_self_loops

__all__ = ["MUTAG"]

def one_hot_encoding(input, num_classes):
    return F.one_hot(input, num_classes = num_classes)

class MUTAG(data.Dataset):
    def __init__(self, data_root, fold, split, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(data_root, transform, pre_transform, pre_filter)
        self.split = split
        self.valid_ids = self._split_train_val(fold)
        out = self._read_data()
        data, self.slices, self.sizes = out
        self.data = Data.from_dict(data) if isinstance(data, dict) else data
        self.data_list = []
        self._fill_data_list()

    @property
    def raw_file_names(self):
        return os.listdir(f"{self.root}/raw")
    
    @property
    def processed_file_names(self):
        return ['MUTAG.pt']
    
    def _split_train_val(self, fold=0):
        from utils.file_dealer import json_loader
        js_dict = json_loader(f"{self.root}/MUTAG_splits.json")[fold]
        val_ids = natsort.natsorted(js_dict['test'])
        train_ids = natsort.natsorted(js_dict['model_selection'][0]['validation'] + js_dict['model_selection'][0]['train'])
        if self.split == 'train':
            return train_ids
        elif self.split == 'val':
            return val_ids

    def _read_data(self):
        # Minus 1 to transform the init num from 1 to 0
        edge_idx = read_txt_array(f"{self.root}/raw/MUTAG_A.txt", sep=',', dtype=torch.long).t() - 1
        batch = read_txt_array(f"{self.root}/raw/MUTAG_graph_indicator.txt", sep=',', dtype=torch.long) - 1
        node_labels = read_txt_array(f"{self.root}/raw/MUTAG_node_labels.txt", sep=',', dtype=torch.long)
        edge_labels = read_txt_array(f"{self.root}/raw/MUTAG_edge_labels.txt", sep=',', dtype=torch.long)
        # Get the ground truth
        graph_labels = read_txt_array(f"{self.root}/raw/MUTAG_graph_labels.txt", sep=',', dtype=torch.long)

        node_labels = one_hot_encoding(node_labels, len(torch.unique(node_labels)))
        edge_labels = one_hot_encoding(edge_labels, len(torch.unique(edge_labels)))

        # Turn -1 to 0
        _, graph_labels = graph_labels.unique(sorted=True, return_inverse=True)
            
        num_nodes = node_labels.size(0)
        edge_idx, edge_attr = remove_self_loops(edge_idx, edge_labels)
        edge_idx, edge_attr = coalesce(edge_idx, edge_labels, num_nodes)
            
        g_data = Data(x=node_labels, edge_index=edge_idx, edge_attr=edge_attr, y=graph_labels)
        g_data, slices = split(g_data, batch)

        sizes = {
            'num_graphs': graph_labels.size(-1),
            'num_node_labels': node_labels.size(-1),
            'num_edge_attributes': edge_attr.size(-1),
            'num_edge_labels': edge_labels.size(-1),
        }
        return g_data, slices, sizes
    
    def _fill_data_list(self):
        for idx in self.valid_ids:
            data = separate(
                cls=self.data.__class__,
                batch=self.data,
                idx=idx,
                slice_dict=self.slices,
                decrement=False,
            )
            self.data_list.append(data)

    def process(self):
        pass

    def len(self):
        return len(self.valid_ids)

    def get(self, idx):
        return self.data_list[idx], self.data_list[idx].y[0]