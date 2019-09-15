# modelnet issue
import os.path as osp

import networkx as nx
import numpy as np
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge


def torch_geometric_2nx(dataset, print_flag=False):
    """
    :param dataset: a list of pytorch_geometric.dataset
    :param print_flag:
    :return:
    """

    graphs, labels = [], []

    for i in range(len(dataset)):
        edges = dataset[i].edge_index  # tensor of shape [2, n_edges * 2]
        n_edge = edges.shape[1] // 2
        edges = edges.numpy()  # edges = np.array([[ 0,  0,  1,  1 ], [ 1,  9,  0,  2]])
        edges_lis = list(edges.T)
        edge_lis_ = set([edge[0] for edge in edges_lis] + [edge[1] for edge in edges_lis])
        edges_lis = [(edge[0], edge[1]) for edge in edges_lis]
        g = nx.from_edgelist(edges_lis)

        assert len(edge_lis_) == len(g)  # check nx.from_edgelist is right
        assert len(g.edges) == n_edge  # check nx.from_edgelist is right

        graphs.append(g)
        assert len(g.edges) == n_edge
        label = dataset[i].y
        labels.append(int(label))

        if print_flag: print(i)
    return graphs, labels


if __name__ == '__main__':
    version = '10'  # 10 or 40
    idx = 1

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet' + version)
    path = '/home/cai.507/Documents/DeepLearning/ModelNet' + version

    pre_transform, transform = FaceToEdge(), None

    # train_dataset = ModelNet(path,path version, True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet(path, version, False, transform=transform, pre_transform=pre_transform)
    data = test_dataset  # train_dataset + test_dataset

    test_data = data[idx]

    data = FaceToEdge()(test_data)
    from torch_geometric.utils import to_networkx
    print(data)
    print(f'num of nodes is {data.pos.shape[0]}')
    g = to_networkx(data, num_nodes=data.pos.shape[0]).to_undirected()
    # gs, _ = torch_geometric_2nx([data])
    # g = gs[0]
    print(nx.info(g))

    # number of connected components
    size_list = [len(g) for g in nx.connected_components(g)]
    print(size_list)
