import numpy as np
import os.path as osp
import networkx as nx
from torch_geometric.transforms import FaceToEdge
from torch_geometric.datasets import ModelNet
from Esme.graph.dataset.qm9 import  graphs_stat
import torch_geometric.transforms as T
import sys
from Esme.helper.load_graph import component_graphs
from Esme.helper.time import timefunction
import torch
from torch_geometric.utils import to_undirected
from sklearn.neighbors import kneighbors_graph
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix
from Esme.helper.time import timefunction
from Esme.viz.pointcloud import plot3dpts

def modelnet_cat():
    # convert y from pytorch_geometric to labels
    dict = {
     0: 'bathtub',
     1: 'bed',
     2: 'chair',
     3: 'desk',
     4: 'dresser',
     5: 'monitor',
     6: 'night_stand',
     7: 'sofa',
     8: 'table',
     9: 'toilet'}

    count = {0: 156, 1: 615, 2: 989, 3: 286, 4: 286, 5: 565, 6: 286, 7: 780, 8: 492, 9: 444}
    return dict

@timefunction
def torch_geometric_2nx(dataset, labels_only = False, print_flag = False, weight_flag = False):
    """
    :param dataset:
    :param labels_only: return labels only
    :param print_flag:
    :param weight_flag: whether computing distance as weights or not
    :return:
    """
    import numpy as np
    graphs, labels = [], []

    if labels_only:
        for i in range(len(dataset)):
            labels.append(int(dataset[i].y[0]))
        return None, labels

    for i in range(len(dataset)):
        edges = dataset[i].edge_index # tensor of shape [2, n_edges * 2]
        n_edge = edges.shape[1] // 2
        edges = edges.numpy() # edges = np.array([[ 0,  0,  1,  1 ], [ 1,  9,  0,  2]])
        edges_lis = list(edges.T)
        edge_lis_ = set([edge[0] for edge in edges_lis] + [edge[1] for edge in edges_lis])
        edges_lis = [(edge[0], edge[1]) for edge in edges_lis]
        g = nx.from_edgelist(edges_lis)

        assert len(edge_lis_) == len(g) # check nx.from_edgelist is right
        assert len(g.edges) == n_edge # check nx.from_edgelist is right

        if weight_flag:
            # set the pos for graph
            pos = dataset[i].pos.numpy()
            assert max(g.nodes()) <= pos.shape[0]

            try:
                assert pos.shape == (len(g), 3)
            except:
                print(f'Isolated pts. pos shape {pos.shape}, n_node is {len(g)}') # todo: need to check with pytorch_geo

            pos_dict = dict()
            for k in g.nodes():
                pos_dict[k] = tuple(pos[k, :])
            g = nx.from_edgelist(edges_lis)

            for node, value in pos_dict.items():
                g.node[node]['pos'] = value

            # compute the edge weight
            for u,v in g.edges():
                pos1, pos2 = g.node[u]['pos'], g.node[v]['pos']
                pos1, pos2 = np.array(pos1), np.array(pos2)
                g[u][v]['dist'] = np.linalg.norm(pos1 - pos2)

        graphs.append(g)
        assert len(g.edges) == n_edge
        label = dataset[i].y
        labels.append(int(label))

        if print_flag: print(i)
    return graphs, labels

def load_modelnet(version='10', point_flag = False):
    """
    :param point_flag: Sample points if point_flag true. Otherwise load mesh
    :return: train_dataset, test_dataset
    """
    assert version in ['10', '40']
    if point_flag:
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    else:
        pre_transform, transform = FaceToEdge(), None

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet' + version)
    path = '/home/cai.507/Documents/DeepLearning/ModelNet' + version

    train_dataset = ModelNet(path, version, True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet(path, version, False, transform=transform, pre_transform=pre_transform)
    return train_dataset, test_dataset

@timefunction
def modelnet2pts2gs(version='10', nbr_size = 8, exp_flag = True, label_only = False, a=None, b=None):
    """ sample points and create neighborhoold graph
    """

    train_dataset, test_dataset = load_modelnet(version=version, point_flag=True)
    all_dataset = train_dataset + test_dataset
    labels = [int(data.y) for data in all_dataset]
    n = len(all_dataset)
    if label_only: return None, labels

    if a is not None and b is not None:
        search_range = range(a,b)
    else:
        search_range = range(n)

    gs = []
    for i in search_range:
        pos = all_dataset[i].pos.numpy() # (1024, 3)
        adj = kneighbors_graph(pos, nbr_size, mode='distance', n_jobs=-1)
        g = nx.from_scipy_sparse_matrix(adj, edge_attribute= 'weight')
        if exp_flag:
            for u, v in g.edges():
                g[u][v]['weight'] = np.exp(-g[u][v]['weight'])
        if i % 10==0: print(f'finish converting {i}')
        gs.append(g)
    return gs, labels

    # lap = normalized_laplacian_matrix(g, weight='w').toarray()
    # np.set_printoptions(threshold=np.nan)
    # print(lap)
    # print(lap[1,:])
    # print(g[1023])

@timefunction
def modelnet2graphs(version = '10', print_flag = False, labels_only = False, a = 0, b = 10, weight_flag = False):
    """ load modelnet 10 or 40 and convert to graphs"""

    train_dataset, test_dataset = load_modelnet(version, point_flag = False)
    all_dataset = train_dataset + test_dataset
    n = len(all_dataset)

    if labels_only:
        labels = []
        for i in range(n):
            lab = int(all_dataset[i].y)
            labels.append(lab)
        return None, labels

    datasets = []
    assert b < n

    for i in range(a,b):
        tmp = FaceToEdge()(all_dataset[i]) if version=='10' else all_dataset[i] # TODO a bug here. should report to pytorch geometric
        datasets.append(tmp)
        if (i % 10==0 and print_flag): print(i)

    graphs, labels = torch_geometric_2nx(datasets, print_flag=print_flag, weight_flag= weight_flag)
    assert len(graphs) == b - a
    return graphs, labels

def modelnet2points(version, idx = 1):
    train_dataset, test_dataset = load_modelnet(version, point_flag=True)
    print(train_dataset[1]) # Data(pos=[1024, 3], y=[1])

    x = train_dataset[idx].pos.numpy()
    plot3dpts(x)


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--test_size", default=20, type=int, help='test size')
parser.add_argument("--w", action='store_true', help='use weight or not')
parser.add_argument("--idx", default=5, type=int, help='index') # only 335 nodes

if __name__ == '__main__':
    # viz 3d graph
    args = parser.parse_args()
    idx = args.idx
    version = '10'
    gs = modelnet2pts2gs(version, exp_flag=True, a = 1, b = 20)

    sys.exit()

    train_dataset, test_dataset = load_modelnet(version, point_flag=False)
    print(len(test_dataset[idx].pos))

    data = FaceToEdge()(test_dataset[idx]) if version=='10' else test_dataset[idx]
    print(data)
    gs, _ = torch_geometric_2nx([data], weight_flag=args.w)
    g = component_graphs(gs[0])[0]
    from Esme.dgms.fil import nodefeat # do to move up. otherwise will have circular import with fil.py
    print(nodefeat(g, 'fiedler')[:5])

    sys.exit()
    from Esme.viz.graph import viz_mesh_graph, network_plot_3D

    edge_index, pos = data.edge_index.numpy(), data.pos.numpy()
    g = viz_mesh_graph(edge_index, pos, viz_flag=False)  # generate_random_3Dgraph(n_nodes=n, radius=0.25, seed=1)
    components_size = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    print(components_size)

    sys.exit()
    network_plot_3D(g, 0, save=False)

    # viz 3d mesh
    idx = 10
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
