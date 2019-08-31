import os.path as osp
import sys

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
import numpy as np
import networkx as nx

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

def torch_geometric_2nx(dataset, print_flag = False):
    graphs, labels = [], []
    for i in range(len(dataset)):
        edges = dataset[i].edge_index # tensor of shape [2, n_edges * 2]
        edges = np.array(edges)
        # edges = np.array([[ 0,  0,  1,  1 ], [ 1,  9,  0,  2]])
        edges_lis = list(edges.T)
        edges_lis = [(edge[0], edge[1]) for edge in edges_lis]
        g  = nx.from_edgelist(edges_lis)
        graphs.append(g)
        labels.append(int(dataset[i].y[0]))
        if print_flag: print(i)
    return graphs, labels

def graphs_stat(graphs):
    print('graphs is of length %s' % len(graphs))
    length_list = list(map(len, graphs))
    edge_len_list = [len(graph.edges()) for graph in graphs]
    m = np.mean(length_list)
    n = np.mean(edge_len_list)
    print(f'The average number of nodes is {m} and average number of edges is {n}')



if __name__ == '__main__':
    target = 0
    dim = 64

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform).shuffle()
    graphs, labels = torch_geometric_2nx(dataset)
    graphs_stat(graphs)


