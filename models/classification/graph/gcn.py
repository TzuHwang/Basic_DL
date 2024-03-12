import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GCN(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, layer_num, dropout=0):
        super(GCN, self).__init__()
        self.conv_layers = {}
        self.last_layer = layer_num
        for i in range(layer_num):
            if i == 0:
                self.conv_layers[f"conv_{i}"] = {pyg_nn.GCNConv(input_channel_num, 64)}
            else:
                self.conv_layers[f"conv_{i}"] = {pyg_nn.GCNConv(64, 64),}
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(64, output_channel_num)

    def forward(self, graph):
        """
        Arguments:
            x: [num_nodes, 7], node features
            edge_index: [2, num_edges], edges
            batch: [num_nodes], batch assignment vector which maps each node to its 
                   respective graph in the batch

        Outputs:
            probs: probabilities of shape (batch_size, output_channel_num)
        """
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        for layer in self.conv_layers:
            x = self.conv_layers[layer](x, edge_index)
            x = self.relu(x) if layer != f"conv_{self.last_layer}" else pyg_nn.global_mean_pool(x, batch)
            x = self.dropout(x)
        x = self.linear(x)
        return x