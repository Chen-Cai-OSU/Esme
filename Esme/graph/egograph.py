import os
import time
from collections import Counter

import networkx as nx
import numpy as np
from joblib import Parallel, delayed

from Esme.applications.motif.aux import add_edge_val
from Esme.graph.generativemodel import sbm
from Esme.helper.format import precision_format
from Esme.helper.io_related import io
from Esme.helper.time import timefunction

class square_class:
    def square_int(self, i):
        return i * i

    def run(self, num):
        results = []
        results = Parallel(n_jobs=-1, backend="threading") \
            (delayed(unwrap_self)(i) for i in zip([self] * len(num), num))
        print(results)

def unwrap_self(*arg, **kwarg):
    return egograph.egograph(*arg, **kwarg)

def unwrap_egograph_batch(*arg, **kwarg):
    return egograph.batch_egograph(*arg, **kwarg)

def split(a, n):
    k, m = divmod(len(a), n)
    tmp = list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))
    res = []
    for r in tmp:
        res.append((min(r), max(r)))
    return res

def merge_tuple(x):
    return (x[0], x[1][0], x[1][1])

# for k in zip(split(range(13), 3), [1,2,3]):
#     print(merge_tuple(k))

def modify_egograph(g, center=1):
    #TODO: undone
    # input: a one-ring egograph with no consecutive labels
    # output: add dist val for nearby nodes
    for n in g.nodes():
        g.node[n]['dist'] = 1
    g.node[center]['dist'] = 0
    for u, v in g.nodes():
        pass

class egograph():

    def __init__(self, g, n = 100, radius = 1, recompute_flag = False, norm_flag=False, delete_center = False, print_flag = True):
        self.graph = g
        self.radius = radius
        self.recompute_flag = recompute_flag
        self.n = n
        self.dataset = ''
        self.norm_flag = norm_flag
        self.delete_center = delete_center
        self.print_flag = print_flag

    def egograph(self, node, delete_center= False):
        t0 = time.time()
        res = nx.ego_graph(self.graph, node, radius=self.radius)
        if self.delete_center: res.remove_node(node)
        if self.print_flag:
            print('Finish node %s in %s' % (node, precision_format(time.time() - t0, 3)))
        return res

    def batch_egograph(self, i, j):
        res = []
        assert max(i,j) < self.n
        for node in range(i, j+1):
            t0 = time.time()
            tmp = nx.ego_graph(self.graph, node, radius=self.radius)
            print('Finish node %s in %s' % (node, time.time() - t0))
            res.append(tmp)
        return res

    def emb_file(self, dataset):
        self.dataset = dataset
        dir = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/EigenPro2/emb', self.dataset, '')
        tmp = 'norm' if self.norm_flag else 'nn'
        file = 'egographs' + str(self.radius) + '_' + str(self.n) + '_' + tmp  # egographs_1_20_norm
        return dir, file

    def compute(self, method = 'serial', print_flag = True):
        dir = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/EigenPro2/emb', self.dataset, '')
        tmp = 'norm' if self.norm_flag else 'nn'
        file =  'egographs' + str(self.radius) + '_' + str(self.n) + '_' + tmp # egographs_1_20_norm
        if self.recompute_flag: os.remove(dir + file) if os.path.exists(dir + file) else 'File do not exist'

        try:
            raise IOError
            # emdio = io(dir, file, saver='pickle')
            # egographs = emdio.load_obj()
            # print(egographs)
            # return egographs
        except IOError:
            t0 = time.time()
            if method == 'serial':
                res = []
                for node in range(self.n):
                    t_ = time.time()
                    g_ = nx.ego_graph(self.graph, node, radius=self.radius)
                    if node % 10 == 0: print('Serial: Finish node %s, takes %s'%(node, time.time()-t_))
                    res.append(g_)
                    # print('res so far takes %s memeory' %sys.getsizeof(res))
                print('Serial version: Finish computing egographs in %s' % (time.time() - t0))

            elif method == 'multithreading':
                res = Parallel(n_jobs=-1, backend='threading', verbose=5)(delayed(unwrap_self)(*i) for i in zip([self] * self.n, range(self.n)))
                print('Multishreading version: Finish computing egographs in %s' % (time.time() - t0))

            elif method == 'parallel':
                res = Parallel(n_jobs=-1, verbose=5)(delayed(unwrap_self)(*i) for i in zip([self] * self.n, range(self.n)))
                print('Parallel version: Finish computing egographs in %s'% (time.time() - t0))
            elif method == 'batch_parallel':
                iter = zip([self]*40, split(range(self.n), 40))
                res_ = Parallel(n_jobs=-1, verbose=5)( delayed(unwrap_egograph_batch)(*merge_tuple(i)) for i in iter) # a list of lists of networks
                res = [] # merge all results
                for i in res_:
                    res += i
                print('Batch Parallel version: Finish computing egographs in %s' % (time.time() - t0))
            else:
                raise Exception('No such method')
            return res

    def egographs(self, method = 'serial'):
        dir, file = self.emb_file(self.dataset)
        emdio = io(dir, file, saver='pickle')

        if self.recompute_flag:
            emdio.rm_obj()

        res = emdio.load_obj()
        if res == 0: # fail to load
            res = self.compute(method=method)
            emdio.save_obj(res)
        return res

def test_egoclass():
    g = nx.random_geometric_graph(200, 0.5)
    ego = egograph(g, radius=1)
    ego.emb_file(dataset='test', radius = 1, n = 101)
    res_parallel = ego.compute(method='multithreading')
    res_serial = ego.compute(method='serial')

def randomize_egographs(egographs):
    # for each egograph, re-generate fv value as random variables. Further destroy the structure.
    tmp_egographs = []
    for g_ in egographs:
        for n in g_.nodes():
            g_.node[n]['fv'] = np.random.rand()
        add_edge_val(g_, edge_value='max')
        tmp_egographs.append(g_)
    return tmp_egographs

@timefunction
def egofeat(g, center = 1):
    t0 = time.time()
    g = nx.ego_graph(g, center, radius=1) # slow quite slow for large graph
    t1 = time.time()

    deg = nx.degree(g, center)
    g.remove_node(center)
    nbr_length = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    n_singleton = Counter(nbr_length)['1']
    n_nonsingle = len(nbr_length) - n_singleton
    n_sum_nonsingle = sum(nbr_length) - n_singleton

    feat01 = n_singleton
    feat015 = n_nonsingle
    feat115= n_sum_nonsingle - n_nonsingle # homology 0
    feat115_1 = len(g.edges()) - deg - n_sum_nonsingle + n_nonsingle
    return np.array((deg, feat01, feat015, feat115, feat115_1))

if __name__=='__main__':
    g, _ = sbm(2000, 0.02)
    for center in range(1, 3):
        print(egofeat(g, center=center))