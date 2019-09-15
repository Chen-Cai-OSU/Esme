""" laplacian eigenmap """

import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg
import time
import sys
from ..helper.format import precision_format

class LaplacianEigenmaps():

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the LaplacianEigenmaps class

        Args:
            d: dimension of the embedding
        '''
        hyper_params = {
            '_method_name': 'lap_eigmap_svd'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, weight=None, print_flag = False):
        graph = graph.to_undirected()
        t0 = time.time()
        L_sym = nx.normalized_laplacian_matrix(graph, weight=weight)
        w, v = lg.eigs(L_sym, k=self._d + 1, which='SM')
        self.eigval = w
        self._X = v[:, 1:]
        if print_flag: print('Compute eigenmaps takes %s'%(precision_format(time.time() - t0)))
        return self._X

        p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        eig_err = np.linalg.norm(p_d_p_t - L_sym)
        if logger== None:
            print('Laplacian matrix recon. error (low rank): %f' % eig_err)
        else:
            logger.info('Laplacian matrix recon. error (low rank): %f' % eig_err)
        return self._X, (t2 - t1)

    def clustering_coefficient(self, g):
        cc = list(nx.closeness_centrality(g).values())
        return np.array(cc).reshape(len(g), 1)

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.exp(-np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2))

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r

    def save_emb(self, dir, file):
        emb = self._X
        np.save(dir + file, emb)
        print('Finish saving embedding at %s'%(dir+file))


if __name__ == '__main__':
    # load Zachary's Karate graph
    G = nx.random_geometric_graph(8000, 0.2)
    print(nx.info(G))
    for u, v in G.edges():
        G[u][v] = {'w': np.random.random()}
    embedding = LaplacianEigenmaps(d=50)
    emd = embedding.learn_embedding(graph=G, weight='w')
    print(np.shape(emd))
    sys.exit()
