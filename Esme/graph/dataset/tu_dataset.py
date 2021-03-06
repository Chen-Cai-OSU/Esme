import os
import shutil
import os.path as osp
import sys
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import numpy as np
import networkx as nx
import argparse
from Esme.dgms.fil import gs2dgms
from Esme.dgms.stats import dgms_summary
from Esme.graph.dataset.modelnet import modelnet2graphs

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_tu_data
from Esme.graph.generativemodel import sbms
class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False):
        self.name = name
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0

        for i in range(self.data.x.size(1)):
            if self.data.x[:, i:].sum().item() == self.data.x.size(0):
                return self.data.x.size(1) - i

        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0

        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

def torch_geometric_2nx_(dataset):
    graphs = []
    for i in range(len(dataset)):
        adj = dataset[i].adj # tensor
        adj = np.array(adj)
        try:
            assert (adj == adj.T).all()
        except AttributeError:
            raise Exception('Non symmetric adj matrix for data %s'%i)
        g = nx.from_numpy_array(adj)
        graphs.append(g)
    return graphs

def torch_geometric_2nx(dataset, labels_only = False):
    graphs, labels = [], []

    if labels_only:
        for i in range(len(dataset)):
            labels.append(int(dataset[i].y[0]))
        return None, labels

    for i in range(len(dataset)):
        edges = dataset[i].edge_index # tensor of shape [2, n_edges * 2]
        edges = np.array(edges)
        # edges = np.array([[ 0,  0,  1,  1 ], [ 1,  9,  0,  2]])
        edges_lis = list(edges.T)
        edges_lis = [(edge[0], edge[1]) for edge in edges_lis]
        g  = nx.from_edgelist(edges_lis)
        graphs.append(g)
        labels.append(int(dataset[i].y[0]))
    return graphs, labels

def graphs_stat(graphs, verbose = 0):
    print('graphs is of length %s' % len(graphs))
    length_list = list(map(len, graphs))
    edge_length_list = [len(g.edges()) for g in graphs]
    m = np.mean(length_list)
    print('The average number of nodes is %s'%m)
    print('The average number of edges is %s' % np.mean(edge_length_list))
    if verbose==1:
        print(edge_length_list)

def load_tugraphs(graph='mutag', labels_only = False):

    # load synthetic sbm graphs
    if graph[:3] == 'syn':
        scale = int(graph[3:])
        n = 750
        p, q = 0.5, 0.1
        p_, q_ = 0.4, 0.2
        gs1 = sbms(n=n, n1=100 * scale, n2=50 * scale, p=p, q=q)
        gs2 = sbms(n=n, n1=75 * scale, n2=75 * scale, p=p, q=q)
        graphs = gs2 + gs1
        labels = [1] * n + [2] * n
        return graphs, labels

    graph = name_conversion(graph)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graph)
    dataset = TUDataset(path, name=graph)
    graphs, labels = torch_geometric_2nx(dataset, labels_only)
    return graphs, labels

def load_shapegraphs(graph = 'mn10'):
    if graph=='mn10':
        graphs, labels = modelnet2graphs(version='10', print_flag=True)
    elif graph=='mn40':
        graphs, labels = modelnet2graphs(version='40', print_flag = True)
    else:
        raise Exception(f'No such graph {graph}')
    return graphs, labels

def name_conversion(graph='mutag'):

    # change mutag to MUTAG. reddit_5K to REDDIT-MULTI-5K
    if graph in ['mutag', 'nci1', 'nci109', 'collab', 'cox2', 'dhfr', 'frankenstein','bzr', 'enzymes']:
        return graph.upper()
    elif graph == 'reddit_5K':
        return 'REDDIT-MULTI-5K'
    elif graph == 'reddit_binary':
        return 'REDDIT-BINARY'
    elif graph == 'reddit_12K':
        return 'REDDIT-MULTI-12K'
    elif graph == 'imdb_binary':
        return 'IMDB-BINARY'
    elif graph == 'imdb_multi':
        return 'IMDB-MULTI'
    elif graph == 'protein_data':
        return 'PROTEINS'
    elif graph == 'ptc':
        return 'PTC_MR'
    elif graph == 'dd_test':
        return 'DD'
    else:
        raise Exception('No such graph %s'%graph)

def ave_diameter(gs):
    """ average diameter of a list of nx graphs """
    # g = nx.random_geometric_graph(100,0.1)
    # gs = [g] * 10
    gs = [max(nx.connected_component_subgraphs(g), key=len) for g in gs]
    diameters = [nx.diameter(g) for g in gs]
    return {'mean': np.mean(diameters), 'std': np.std(diameters)}

def diameter_stats():
    res ={'cox2': {'mean': 13.790149892933618, 'std': 1.0382767487921656}, 'nci109': {'mean': 13.123334141022534, 'std': 5.308880028163508}, 'dhfr': {'mean': 14.603174603174603, 'std': 2.582232838035949}, 'frankenstein': {'mean': 8.474752132810698, 'std': 4.288572881190721}, 'imdb_binary': {'mean': 1.861, 'std': 0.3459465276599839}, 'imdb_multi': {'mean': 1.474, 'std': 0.49932354240512244}, 'mutag': {'mean': 8.21808510638298, 'std': 1.8422617208410026}, 'ptc': {'mean': 7.523255813953488, 'std': 4.586321280508768}, 'nci1': {'mean': 13.309489051094891, 'std': 5.5376246791257975}, 'dd_test': {'mean': 19.8981324278438, 'std': 7.718067787093092}}

    return res


parser = argparse.ArgumentParser()
parser.add_argument('--graph', type=str, default='MUTAG', help="graph dataset")

if __name__ == '__main__':
    modelnet2graphs('10')
    sys.exit()
    diameters = dict()
    for gname in ['cox2', 'nci109',  'dhfr', 'frankenstein','imdb_binary', 'imdb_multi', 'mutag', 'ptc' ,'nci1', 'dd_test', 'reddit_binary', 'reddit_5K',]:
        gs, labels = load_tugraphs(gname)
        diameters[gname] = ave_diameter(gs)
        print(diameters)


    sys.exit()
    print(len(gs))
    # sys.argv = []
    max_nodes = 30000
    args = parser.parse_args()
    print(args)
    class MyFilter(object):
        def __call__(self, data):
            return data.num_nodes <= max_nodes

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.graph) # path = '/home/cai.507/Documents/DeepLearning/Esmé/Esme/graph/dataset/../data/MUTAG'
    dataset = TUDataset(path, name=args.graph) # dataset = TUDataset(path, name=args.graph, transform=T.ToDense(max_nodes))

    graphs, labels = torch_geometric_2nx(dataset)
    graphs_stat(graphs)

    subdgms = gs2dgms(graphs, n_jobs=1, fil='deg', fil_d='sub', one_hom=False, debug_flag=True)
    dgms_summary(subdgms)
