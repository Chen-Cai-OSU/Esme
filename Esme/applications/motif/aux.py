import numpy as np
import time
import networkx as nx
import os

from Esme.dgms.vector import pdvector
from Esme.graph.function import function_basis
from Esme.helper.format import precision_format
from Esme.helper.io import io, make_dir
from Esme.dgms.kernel import sw_parallel

def n_node(dataset):
    if dataset=='wikipedia':
        return 4777
    elif dataset == 'blogcatalog':
        return 10312
    elif dataset == 'flickr':
        return 80513
    else:
        print('No such dataset') #TODO fix this. add exception back
        return 0
        # raise Exception('No such dataset')

def pdemb(dgms, dataset='wikipedia', recompute_flag=True, norm=True):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/EigenPro2/emb/' + dataset + '/'
    make_dir(direct)
    tmp = 'norm' if norm else 'nn'
    file = direct + 'pd_vector_' + tmp
    file_ = file + '.npy'
    n = n_node(dataset)
    if recompute_flag: os.remove(file_) if os.path.exists(file_) else 'File do not exist'

    try:
        pdvectors = np.load(file_)
        print('Loading existing persistence embedding from %s'%file_)
    except IOError:
        pdv = pdvector()
        pdvectors = pdv.persistence_vectors(dgms)
        np.save(file, pdvectors)
        print('Finish saving %s pdvectors at %s'%(n, file))
    return pdvectors

def add_edge_val(gi, edge_value='max'): # only works for sublevel filtration
    for (e1, e2) in gi.edges():
        if edge_value == 'max':
            gi[e1][e2]['fv'] = max(gi.node[e1]['fv'], gi.node[e2]['fv'])
        if edge_value == 'min':
            gi[e1][e2]['fv'] = min(gi.node[e1]['fv'], gi.node[e2]['fv'])

def network2g_(graph):
    from helper.io import io

    """ convert a network to a graph with filtration function, ready to compute PD"""
    # graph = nx.random_geometric_graph(5000, 0.1)
    t0 = time.time()
    print('Start converting network2g')
    g = nx.convert_node_labels_to_integers(graph, first_label=0)
    assert len(g) == max(g.nodes()) + 1
    g = function_basis(g, ['deg'], norm_flag='no')

    for n in g:
        g.node[n]['fv'] = g.node[n]['deg']
    add_edge_val(g, edge_value='max')
    print('Finish preparing network for egographs step, takes %s'%(precision_format(time.time()-t0)))
    return g

class network2g():
    def __init__(self, fil = 'deg', norm_flag='no', sub = True, dataset = 'blogcatalog', recomp = False):
        self.dir = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/EigenPro2/emb', dataset,'')
        make_dir(self.dir)
        self.fil = fil
        self.norm_flag = norm_flag
        self.sub = sub
        self.dataset = dataset
        self.file = dataset + '_' + fil + '_' + norm_flag # TODO incorporate sub/super
        self.io = io(self.dir, self.file, saver='pickle')
        self.recomp = recomp
        print(self.dir + self.file)

    def compute(self, graph):
        """ convert a network to a graph with filtration function, ready to compute PD"""
        # graph = nx.random_geometric_graph(5000, 0.1)

        res = self.io.load_obj()
        if res!=0 and not self.recomp: return res

        t0 = time.time()
        print('Start converting network2g')
        g = nx.convert_node_labels_to_integers(graph, first_label=0)
        g.remove_edges_from(g.selfloop_edges())
        assert len(g) == max(g.nodes()) + 1
        g = function_basis(g, [self.fil], norm_flag=self.norm_flag)
        for n in g.nodes():
            g.nodes[n]['fv'] = g.nodes[n][self.fil]
        add_edge_val(g, edge_value='max')
        print('Finish preparing network for egographs step, takes %s' % (precision_format(time.time() - t0)))
        self.io.save_obj(g)
        return g

def nc_prework(dataset, norm_flag=True, feat='pd_vector'):
    # set the dir for nc classification script
    mat_f = os.path.join('/home/cai.507/Documents/DeepLearning/EmbeddingEval/data/', dataset + '.mat')
    dir = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/EigenPro2/emb/', dataset, '')
    tmp = 'norm' if norm_flag else 'nn'
    if feat == 'pd_vector':
        file = dir + 'pd_vector_' + tmp
    elif feat =='lap_wg':
        file = dir + 'lap_wg'
    elif feat == 'lap_uwg':
        file = dir + 'lap_uwg'
    else:
        raise Exception('No such feat %s'%feat)
    return mat_f, dir, file

def sparse2graph(x):
    from collections import defaultdict
    from six import iteritems

    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k, v in iteritems(G)}

def load_labels(matfile, adj_matrix_name, label_matrix_name):
    from scipy.io import loadmat
    import scipy as sp
    from sklearn.preprocessing import MultiLabelBinarizer

    mat = loadmat(matfile)
    labels_matrix = mat[label_matrix_name]
    labels_sum = labels_matrix.sum(axis=1)
    indices = np.where(labels_sum > 0)[0]
    labels_matrix = sp.sparse.csc_matrix(labels_matrix[indices])
    A = mat[adj_matrix_name][indices][:, indices]
    graph = sparse2graph(A)

    labels_count = labels_matrix.shape[1]
    mlb = MultiLabelBinarizer(range(labels_count))
    return mat, A, graph, labels_matrix, labels_count, mlb, indices

def wadj(g, swdgms, n = 50, **kwargs):
    """ Compute the weighted adj matrix from dgm-similarity(based on sw) """
    sw_kernel, _ = sw_parallel(swdgms[:n], swdgms[:n], kernel_type='sw', parallel_flag=True, **kwargs)
    adj = nx.adjacency_matrix(g)
    adj = np.sign(adj[:n, :n].todense())
    w_adj = np.multiply(adj, sw_kernel)
    print(w_adj)
    assert np.shape(w_adj) == (n, n)
    w_g = nx.from_numpy_matrix(w_adj)
    return w_g # for laplacian eigenmap

