# gnn baseline
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from scipy.stats import moment
from Esme.helper.time import timefunction

def aggregation(nbrvals, dummy_flag = False, d = 2):
    res = {'min': np.min(nbrvals, axis=1), 'max': np.max(nbrvals, axis=1),
           'mean': np.mean(nbrvals, axis=1), 'std': np.std(nbrvals, axis=1),
           'sum': np.sum(nbrvals, axis=1),
           '3rdmom': moment(nbrvals, moment=3, axis=1),
           '4thmom': moment(nbrvals, moment=4, axis=1)}

    if dummy_flag:
        inival = np.zeros(d)
        res = {'min': inival, 'max': inival, 'mean': inival, 'std': inival, 'sum': inival, '3rdmom': inival, '4thmom': inival}
    return res

class gnn_bl():
    def __init__(self, g, d = 5, n_feat = 7):
        self.graph = g
        self.mem = None
        self.n = len(self.graph)
        self.d = d
        for n in self.graph.nodes():
            self.graph.node[n]['fv_0'] = 2 * np.random.rand(self.d, 1) - 1
        self.n_feat = n_feat

    def nbr_feat(self, node = 0, key = 'fv_0'):
        nbrs = nx.neighbors(self.graph, node)
        res = []
        for n in nbrs:
            res.append(self.graph.node[n][key])
        # print(res)
        return res

    @timefunction
    def iter(self, k = 1, key='fv_0'):
        for n in self.graph.nodes():
            if nx.degree(self.graph, n) > 0:
                nbrvals = np.hstack(self.nbr_feat(n, key=key))
                tmp = aggregation(nbrvals, d = self.d)
            else:
                tmp = aggregation(nbrvals, d = self.d, dummy_flag=True)
            self.graph.node[n]['fv_'+ str(k) + '_'] = tmp
            self.graph.node[n]['fv_' + str(k)] = np.hstack(tmp.values()).reshape(self.n_feat * self.d, 1)

        # iter 2
        for n in self.graph.nodes():
            if nx.degree(self.graph, n) > 0:
                nbrvals = np.hstack(self.nbr_feat(n, key='fv_1'))
                tmp = aggregation(nbrvals)
            else:
                tmp = aggregation(nbrvals, d=self.n_feat * self.d, dummy_flag=True)
            self.graph.node[n]['fv_'+ str(2) + '_'] = tmp
            self.graph.node[n]['fv_' + str(2)] = np.hstack(tmp.values()).reshape(self.n_feat**2 * self.d, 1)

        # iter 3
        for n in self.graph.nodes():
            if nx.degree(self.graph, n) > 0:
                nbrvals = np.hstack(self.nbr_feat(n, key='fv_2'))
                tmp = aggregation(nbrvals)
            else:
                tmp = aggregation(nbrvals, d=self.n_feat**2 * self.d, dummy_flag=True)
            self.graph.node[n]['fv_'+ str(3) + '_'] = tmp
            self.graph.node[n]['fv_' + str(3)] = np.hstack(tmp.values()).reshape(self.n_feat**3 * self.d, 1)

        return self.graph

    def feat(self):
        self.iter()
        x = np.zeros((self.n, (self.n_feat**2 + self.n_feat + 1) * self.d))
        for n in self.graph.nodes():
            tmp = self.graph.node[n]
            feat1 = np.concatenate((tmp['fv_0'].reshape(1, self.d),
                                   tmp['fv_1_']['mean'].reshape(1, self.d),
                                   tmp['fv_1_']['max'].reshape(1, self.d),
                                   tmp['fv_1_']['min'].reshape(1, self.d),
                                   tmp['fv_1_']['std'].reshape(1, self.d),
                                   tmp['fv_1_']['sum'].reshape(1, self.d),
                                   tmp['fv_1_']['3rdmom'].reshape(1, self.d),
                                   tmp['fv_1_']['4thmom'].reshape(1, self.d)),
                                  axis = 1)

            feat2 = np.concatenate((
                                   tmp['fv_2_']['mean'].reshape(1, self.d * self.n_feat),
                                   tmp['fv_2_']['max'].reshape(1, self.d * self.n_feat),
                                   tmp['fv_2_']['min'].reshape(1, self.d * self.n_feat),
                                   tmp['fv_2_']['std'].reshape(1, self.d * self.n_feat),
                                   tmp['fv_2_']['sum'].reshape(1, self.d * self.n_feat),
                                   tmp['fv_2_']['3rdmom'].reshape(1, self.d * self.n_feat),
                                   tmp['fv_2_']['4thmom'].reshape(1, self.d * self.n_feat)),
                                  axis = 1)

            # feat3 = np.concatenate((
            #     tmp['fv_3_']['mean'].reshape(1, self.d * self.n_feat ** 2),
            #     tmp['fv_3_']['max'].reshape(1, self.d * self.n_feat ** 2),
            #     tmp['fv_3_']['min'].reshape(1, self.d * self.n_feat ** 2),
            #     tmp['fv_3_']['std'].reshape(1, self.d * self.n_feat ** 2),
            #     tmp['fv_3_']['sum'].reshape(1, self.d * self.n_feat ** 2),
            #     tmp['fv_3_']['3rdmom'].reshape(1, self.d * self.n_feat ** 2),
            #     tmp['fv_3_']['4thmom'].reshape(1, self.d * self.n_feat ** 2)),
            #     axis=1)

            x[n] = np.concatenate((feat1, feat2), axis=1)
        # return x
        return normalize(x, axis=1)

if __name__=='__main__':
    g = nx.random_geometric_graph(100, 0.1)
    model = gnn_bl(g, d = 2)
    v = model.feat()
    print(v.shape)


