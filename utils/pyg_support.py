from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
EPS = 1e-15

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.explain import Explainer, PGExplainer, GNNExplainer, CaptumExplainer

from utils.tb_lib import colors

'''
ref: https://e0hyl.github.io/BLOG-OF-E0/GNNExplainer/
'''

class Draw_graph:
    def __init__(self, graph, target, explaination=None, label_rep={0: '0'}, path=None):
        self.path = path

        self.x = graph.x
        assert(self.x.shape[-1]<len(colors))
        self.node_att = torch.argmax(self.x, 1)
        self.label_rep = label_rep
        self.edge_index = graph.edge_index
        self.target = target
        if explaination is not None:
            assert(torch.equal(graph.edge_index, explaination.edge_index))
            self.edge_att = explaination.edge_mask
        else:
            self.edge_att = torch.ones(graph.edge_index.shape[-1])
        self.minv = torch.min(self.edge_att[self.edge_att!=0])*0.1
        self._visualize_graph()

    def _visualize_graph(self):
        G = nx.DiGraph()
        node_size = 800

        for node in self.edge_index.view(-1).unique().tolist():
            G.add_node(node)

        for (src, dst), w in zip(self.edge_index.t().tolist(), self.edge_att.tolist()):
            G.add_edge(src, dst, alpha=w)

        ax, pos = plt.gca(), nx.spring_layout(G)

        for src, dst, data in G.edges(data=True):
            ax.annotate(
                '',
                xy=pos[src],
                xytext=pos[dst],
                arrowprops=dict(
                    arrowstyle="->",
                    alpha=np.max([data['alpha'], self.minv]),
                    shrinkA=sqrt(node_size) / 2.0,
                    shrinkB=sqrt(node_size) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

        for key in self.label_rep:
            node_list = (self.node_att==key).nonzero(as_tuple=True)[0].tolist()
            color = (np.array(colors[key+1])/255).reshape(1,-1)
            if len(node_list)>0:
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, nodelist=node_list,
                                            node_color=color, margins=0.1, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=10)

        if self.path is not None:
            plt.savefig(self.path)
        else:
            plt.show()

        plt.close()