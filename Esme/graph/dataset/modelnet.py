import os.path as osp
import networkx as nx
from torch_geometric.transforms import FaceToEdge
from torch_geometric.datasets import ModelNet
from Esme.graph.dataset.qm9 import  graphs_stat
import torch_geometric.transforms as T
import sys

import torch
from torch_geometric.utils import to_undirected
import numpy as np
def torch_geometric_2nx(dataset, labels_only = False, print_flag = False):
    graphs, labels = [], []

    if labels_only:
        for i in range(len(dataset)):
            labels.append(int(dataset[i].y[0]))
        return None, labels

    for i in range(len(dataset)):
        edges = dataset[i].edge_index # tensor of shape [2, n_edges * 2]
        n_edge = edges.shape[1] // 2
        edges = np.array(edges)
        # edges = np.array([[ 0,  0,  1,  1 ], [ 1,  9,  0,  2]])
        edges_lis = list(edges.T)
        edges_lis = [(edge[0], edge[1]) for edge in edges_lis]
        g  = nx.from_edgelist(edges_lis)
        graphs.append(g)
        assert len(g.edges) == n_edge
        labels.append(int(dataset[i].y[0]))

        if print_flag: print(i)
    return graphs, labels

from Esme.helper.time import timefunction

def load_modelnet(version='10', point_flag = False):
    """
    :param point_flag: Sample points if point_flag true. Otherwise load mesh
    :return: train_dataset, test_dataset
    """
    assert version in ['10', '40']
    if point_flag:
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(2048)
    else:
        pre_transform, transform = FaceToEdge(), None

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet' + version)
    path = '/home/cai.507/Documents/DeepLearning/ModelNet' + version

    train_dataset = ModelNet(path, version, True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet(path, version, False, transform=transform, pre_transform=pre_transform)
    return train_dataset, test_dataset

@timefunction
def modelnet2graphs(version = '40', print_flag = False, labels_only = False, a = 0, b = 10):
    """ load modelnet 10 or 40 and convert to graphs"""

    train_dataset, test_dataset = load_modelnet(version, point_flag = False)
    all_dataset = train_dataset + test_dataset
    n = len(all_dataset)
    print(test_dataset[10])
    sys.exit()

    if labels_only:
        labels = []
        for i in range(n):
            lab = int(data[i].y)
            labels.append(lab)
        return None, labels

    datasets = []
    assert b < n
    for i in range(a,b):
        tmp = FaceToEdge()(all_dataset[i]) if version=='10' else all_dataset[i] # TODO a bug. should report to pytorch geometric
        datasets.append(tmp)
        if i % 10==0 and print_flag: print(i)

    graphs, labels = torch_geometric_2nx(datasets, print_flag=print_flag)
    return graphs, labels

def modelnet2points(version, idx = 1):
    train_dataset, test_dataset = load_modelnet(version, point_flag=True)
    print(train_dataset[1]) # Data(pos=[1024, 3], y=[1])

    x = train_dataset[idx].pos.numpy()
    from Esme.viz.pointcloud import plot3dpts
    plot3dpts(x)


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--test_size", default=20, type=int, help='test size')

if __name__ == '__main__':
    idx = 100
    train_dataset, test_dataset = load_modelnet('10', point_flag = False)
    data = train_dataset[idx]
    face, pos = data.face.numpy(), data.pos.numpy()
    print(face.shape, pos.shape, type(face))
    from Esme.viz.pointcloud import plot_example
    plot_example(face=face, pos=pos)
    sys.exit()
    args = parser.parse_args()
    version = '10'
    for idx in range(1,1000, 100):
        modelnet2points(version, idx)
    sys.exit()

    graphs, labels = modelnet2graphs(version=version, print_flag=True, labels_only=False, a = 0, b = 10)
    graphs_stat(graphs)

    from collections import Counter
    print(Counter(labels))
    print(labels)
    # sys.exit()
    version = '40'
    path = '/home/cai.507/Documents/DeepLearning/ModelNet10'
    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    pre_transform, transform = FaceToEdge(), None
    train_dataset = ModelNet(path, version, True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet(path, version, False, transform=transform, pre_transform=pre_transform)
    print(train_dataset, train_dataset.shape)
    print(test_dataset, test_dataset.shape)
    sys.exit

    print(type(test_dataset))
    print(dir(test_dataset))
    n = len(train_dataset)
    print(n)

    datasets = []
    for i in range(n):
        tmp = FaceToEdge()(train_dataset[i])
        datasets.append(tmp)
        if i % 10==0: print(i)

    graphs, labels = torch_geometric_2nx(datasets)
    graphs_stat(graphs)
    sys.exit()


    dataset = train_dataset
    i = 10
    data = FaceToEdge()(dataset[i])
    print(data.edge_index)
    # print('data',dir(dataset[i]))
    # print('face',dataset[i].face)
    # print('edge', )
    # edges = dataset[i].edge_index
    sys.exit()
    graphs, labels = torch_geometric_2nx(test_dataset)
    graphs_stat(graphs)
