import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import radius_neighbors_graph

from Esme.viz.graph import viz_graph


def random_insertion(G, n=3, print_flag = False):
    G = G.copy()
    nodes = list(G.nodes)
    n_edges = len(G.edges())
    e = random.sample(nodes, 2)
    e.sort()
    i = 1
    while (i <= n):
        if tuple(e) not in list(G.edges):
            G.add_edge(e[0], e[1])
            i = i + 1
        else:
            e = random.sample(nodes, 2)
            e.sort()
    if print_flag:
        print('Before: %s edges. After: %s'%(n_edges, len(G.edges())))
    return G

def random_deletion(G, n = 3, print_flag = False):
    G = G.copy()
    n_edges = len(G.edges())
    for i in range(n):
        e = random.sample(list(G.edges),1)[0]
        G.remove_edge(e[0],e[1])
    if print_flag:
        print('Before: %s edges. After: %s' % (n_edges, len(G.edges())))
    return G

if __name__=='__main__':

    def gmm(means = [[-10, -10], [1,1]], cov = [[1, 0], [0, 1]]):
        # means = [[0, 0], [1,1]]
        # cov = [[1, 0], [0, 1]]  # diagonal covariance
        x, y = np.array([0]), np.array([0])
        n = 10000
        for mean in means:
            x_, y_ = np.random.multivariate_normal(mean, cov, n).T
            x, y = np.concatenate((x, x_)), np.concatenate((y, y_))
        x, y = x[1:].reshape(2*n, 1), y[1:].reshape(2*n, 1)
        x_train = np.concatenate((x, y), axis=1)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(x_train)
        print('Fitted model parameters: mean %s'%gmm.means_)
        # sample
        # gmm.means_ = mean
        # gmm.covariances_ = cov
        return gmm

    def radius_graph(model, n_sample = 50, radius = 2):
        # n_sample, radius = 500, 2
        print(model)
        print(n_sample)
        x_sample, y_sample = model.sample(n_sample)
        plt.scatter(x_sample[:,0], x_sample[:,1])
        plt.show()
        adj = radius_neighbors_graph(x_sample, radius, n_jobs=-1, mode='distance')
        assert np.max(adj) < radius
        g = nx.from_scipy_sparse_matrix(adj) # neighborhoold graph
        return g

    model = gmm()
    g = radius_graph(model, n_sample=500, radius=1)
    viz_graph(g, show=True, edge_width=0.1)
    deg_dict = dict(nx.degree(g))
    plt.plot(deg_dict.values())
    plt.show()