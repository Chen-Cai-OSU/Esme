import os
from collections import Counter

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix

from Esme.applications.motif.aux import load_labels, pdemb, network2g
from Esme.dgms.compute import alldgms
from Esme.dgms.compute import dgm2diag
from Esme.dgms.compute import graph2dgm
from Esme.graph.egograph import egograph
from Esme.graph.function import function_basis

np.random.seed(42)

def pdgraph(n = 100, r = 3):
    """ regular graph with random filtration. Label birth value as 0, death value as 2 and others as 1 """
    g = nx.random_regular_graph(d=r, n = n)
    if r == 2: g = nx.cycle_graph(n)

    g = function_basis(g, ['random'])
    for u, v in g.edges():
        g[u][v]['random'] = max(g.node[u]['random'], g.node[v]['random'])
    g2dgm = graph2dgm(g)
    dgm = g2dgm.get_diagram(g, key='random')
    diag = np.array(dgm2diag(dgm))
    birth_val, death_val = list(diag[:, 0].astype(float)), list(diag[:, 1].astype(float))

    for n in g.nodes():
        e1 = min(abs(np.array(birth_val) - g.node[n]['random']))
        e2 = min(abs(np.array(death_val) - g.node[n]['random']))
        if e1 < 1e-4:
            g.node[n]['label'] = 0
        elif e2 < 1e-4:
            g.node[n]['label'] = 1
        else:
            g.node[n]['label'] = 2

    res = nx.get_node_attributes(g, 'label')
    print('label stat is ', Counter(res.values()))
    # assert Counter(res.values())[-1] == Counter(res.values())[1] == len(dgm)
    return g

def truelabels(dataset = 'wikipedia'):
    dir = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/'
    network = os.path.join(dir, 'data/' + dataset + '.mat')
    mat, A, _, labels_matrix, _, _, indices = load_labels(network, 'network', 'group')
    graph = nx.from_scipy_sparse_matrix(mat['network'])  # TODO: graph rename
    return graph, labels_matrix

def pdfeat(g, fil='deg', dataset='wikipedia', radius = 1, norm_flag = True, recomp=True):
    n2g = network2g(dataset=dataset, fil=fil, norm_flag='yes', sub=True, recomp=recomp)
    g = n2g.compute(g)
    n_node = len(g)

    ego = egograph(g, radius=radius, n=n_node, recompute_flag=recomp, norm_flag=norm_flag)
    ego.emb_file(dataset=dataset)
    egographs = ego.egographs(method='batch_parallel')
    dgms = alldgms(egographs, radius=radius, dataset=dataset, recompute_flag=recomp, method='serial', n=n_node)  # can do parallel computing
    pdvector = pdemb(dgms, dataset, recompute_flag=True, norm=norm_flag)
    return pdvector

# g = nx.random_geometric_graph(100, 1)
# pdfeat(g, dataset='test')

def load_pddata(n = 1000, r = 3, d=10):
    # g = nx.random_geometric_graph(100, 0.1)
    g = pdgraph(n, r = r)
    labels_ = nx.get_node_attributes(g, 'label')
    labels = np.zeros((len(g), 3))
    for i in range(len(g)):
        labels[i, labels_[i]] = 1

    adj = nx.adjacency_matrix(g)
    features_ = nx.get_node_attributes(g, 'random')
    features = np.zeros((len(g), 1))
    for i in range(len(g)):
        features[i, 0] = features_[i]
    features = np.random.random((len(g), d))  # featureless
    # features = np.zeros((len(g), d))  # featureless
    features = lil_matrix(features)

    mask = np.zeros((3, len(g)))
    idx = np.random.choice([0, 1, 2], len(g), p=[0.5, 0.2, 0.3])
    for i in range(len(g)):
        mask[idx[i], i] = 1
    mask = mask.astype(bool)
    train_mask, val_mask, test_mask = mask[0], mask[1], mask[2]

    y_train, y_val, y_test = np.zeros(labels.shape), np.zeros(labels.shape), np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
