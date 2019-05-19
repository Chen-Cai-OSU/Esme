# calculate the smoothness of signal on graph
from Esme.viz.graph import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import dionysus as d
import sklearn_tda as tda
from scipy.spatial.distance import hamming, jaccard,pdist, squareform, cityblock, pdist, euclidean

def featdist(g, u, v, key = 'label', dist_type=None):
    # compute the feat distance on graph

    if key == 'deg':
        res = abs(g.node[u][key] - g.node[v][key])

    elif key == 'dgm':
        if dist_type == 'bd':
            res = d.bottleneck_distance(g.node[u][key], g.node[v][key])
        elif dist_type == 'sw':
            tda_kernel = tda.SlicedWassersteinKernel(num_directions=10, bandwidth=1)
            x = tda_kernel.fit(g.node[u][key])
            y = tda_kernel.transform(g.node[v][key])
            res = y[0, 0] # get the number from (1, 1) array
        else:
            raise Exception('No such distance %s is defined.'%dist_type)

    elif key == 'label':
        v1, v2 = g.node[u][key], g.node[v][key]
        if dist_type == 'hamming':
            res = hamming(v1, v2)
        elif dist_type == 'jaccard':
            res = jaccard(v1, v2)
        else:
            raise Exception('No such distance %s is defined.' % dist_type)

    else:
        # print('All other keys are treated at vector')
        res = euclidean(g.node[u][key], g.node[v][key])

    return res

def diff(g, key = 'label', dist_type = None, viz_flag = False):
    res = []
    for u,v in g.edges():
        tmp = featdist(g, u, v, key=key, dist_type=dist_type)
        assert type(float(tmp)) in [int, float]
        res.append(tmp)
    res = np.array(res) / float(np.mean(res))

    if viz_flag:
        plt.plot(res, '.', markersize=1)
        plt.show()

    return res


def viz_feat_smoothness(g, labels_matrix, sw_distance):
    # visualize the smoothness of the features(labels, sw_dgm) on graphs

    adj = np.sign(nx.adjacency_matrix(g).todense())

    s1 = pdist(labels_matrix.todense(), metric='hamming')
    s1 = squareform(s1)
    s2 = sw_distance

    musk_s1 = np.multiply(adj, s1)

    # get upper triangle indices
    x_indices_, y_indices_ = np.nonzero(musk_s1)
    x_indices, y_indices = [], []
    for i in range(len(x_indices_)):
        if x_indices_[i] < y_indices_[i]:
            x_indices.append(x_indices_[i])
            y_indices.append(y_indices_[i])
    x_indices, y_indices = np.array(x_indices), np.array(y_indices)

    musk_s1_nnz = np.array(musk_s1[x_indices, y_indices]).reshape(88428,)
    musk_s2 = np.multiply(adj, s2)
    musk_s2_nnz = np.array(musk_s2[x_indices, y_indices]).reshape(88428,)
    plt.plot(musk_s1_nnz / np.mean(musk_s1_nnz), '.', markersize=1)
    plt.plot(musk_s2_nnz / np.mean(musk_s2_nnz), '.', markersize=1)
    plt.show()

if __name__ == '__main__':
    g = nx.random_geometric_graph(100, 0.1)
    for n in g.nodes():
        g.node[n]['label'] = np.random.random()
        # g.node[n]['label2'] = np.random.random((1, 3)) - 1

    featdist(g, 0, 1, key='label', dist_type='hamming')
    diff(g, key='label', dist_type='hamming', viz_flag=True)

