import torch.nn as nn
import torch_geometric.nn as pyg_nn

from utils.tb_lib import translate_pred

class GIN(nn.Module):
    """
    ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = pyg_nn.MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(pyg_nn.GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = pyg_nn.MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=dropout)

    def forward(self, graph, edge_index=None, batch=None, to_prob=False):
        if edge_index is None:
            x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        else:
            x, edge_index, batch = graph, edge_index, batch
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = pyg_nn.global_add_pool(x, batch)
        return self.mlp(x) if to_prob is False else translate_pred(self.mlp(x))