import dionysus as d
import networkx as nx
import numpy as np
from numpy import random
import sys
import json

# sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/aux/')
from Esme.permutation.aux.tools import load_data, dump_data, read_all_graphs, generate_graphs, unit_vector, set_betalist, \
                        beta_name_not_in_allowed,  diag2dgm, unzip_databundle, fv, flip_dgm, add_dgms, print_dgm, dgm2diag, make_direct, n_distinct, normalize_
from Esme.permutation.aux.helper import get_subgraphs, attribute_mean, get_diagram



def fake_diagram(cardinality = 2, seed=42, true_dgm = 'null'):
    random.seed(seed)
    if true_dgm!='null':
        array_tmp = dgm2diag(true_dgm)
        sample_pool = [p[0] for p in array_tmp] + [p[1] for p in array_tmp]
    else:
        raise Exception('true_dgm must not be null')
    try:
        sample = np.random.choice(sample_pool, size=2*cardinality, replace=False)
    except:
        sample = np.random.choice(sample_pool, size=2 * cardinality, replace=True)

    assert set(sample).issubset(set(sample_pool))
    dgm = []
    for i in range(0, len(sample),2):
        x_ = sample[i]
        y_ = sample[i+1]
        dgm.append((min(x_, y_), max(x_, y_)+1e-3))
    return d.Diagram(dgm)

def fake_diagrams(dgms, true_dgms = ['null']*10000, seed=45):
    fake_dgms = []
    for i in range(len(dgms)):
        cardin = len(dgms[i])
        if len(dgms[i])==0:
             fake_dgms.append(d.Diagram([(0,0)]))
             continue
        # print cardin
        tmp_dgm = fake_diagram(cardinality = cardin, seed=seed, true_dgm=true_dgms[i])
        fake_dgms.append(tmp_dgm)
    return fake_dgms


def partial_diagram(dgm, portion=0.5, seed=42+1):
    # dgm = d.Diagram([(1,2)] * 10 + [(3,9)] * 10)
    if portion == 0:
        return dgm
    n = len(dgm)
    diag = dgm2diag(dgm)
    idx_fix = np.random.choice(n, max(int(n * (1 - portion)), 1), replace=False)
    idx_change = set(range(len(diag))) - set(idx_fix)
    diag_fix = [diag[i] for i in range(n) if i in idx_fix]
    dgm_fix = diag2dgm(diag_fix)
    diag_change = [diag[i] for i in range(n) if i in idx_change]
    dgm_change = diag2dgm(diag_change)
    fake_dgm = fake_diagram(cardinality=len(dgm_change), seed=seed, true_dgm=dgm_change)
    return add_dgms(dgm_fix, fake_dgm)

def partial_dgms(dgms, portion=0.5, seed=42+1):
    partial_dgms = []
    for dgm in dgms:
        partial_dgms.append(partial_diagram(dgm, portion=portion, seed=seed))
    return partial_dgms

class gridsearch():

    def __init__(self, dataset = 'mutag', method = 'sw',
                 filtration = 'deg', tf = 'normal', suffix = ''):
        self.method = method
        self.filtration = filtration # deg, ricci, cc...
        self.dataset = dataset # mutag, ptc...
        self.tf = tf # normal, fake...
        self.direct = None
        self.suffix = suffix
        self.one_param = None
        if self.method=='sw':
            # bw = [0.01, 0.1, 1, 10, 100]
            bw = [1, 0.1, 10, 0.01, 100, 1000, 0.001]
            k = [1]
            p=[1]
            self.params = {'bw': bw, 'K': k, 'p': p}
        elif self.method == 'pss':
            bw = [1, 5e-1, 5, 1e-1, 10, 5e-2, 50, 1e-2, 100, 5e-3, 500, 1e-3, 1000]
            k = [1]
            p = [1]
            self.params = {'bw': bw, 'K': k, 'p': p}
        elif self.method == 'wg':
            bw = [1, 10, 0.1, 100, 0.01]
            k = [ 1, 10, 0.1]
            p = [ 1, 10, 0.1]
            self.params = {'bw': bw, 'K': k, 'p': p}
        elif self.method == 'pi':
            res_list = [[50, 50]]
            weight_list = [(1, 1)]
            self.params = {'bandwidth': [1.0, 0.1, 10], 'weight': weight_list,
                           'im_range': [[0, 1, 0, 1]], 'resolution': res_list}
        elif self.method == 'pl':
            num_list = [3, 5]
            res_list = [50, 100, 200]
            self.params= {'num_landscapes': num_list, 'resolution': res_list}
        elif self.method == 'bl0':
            self.params = {}
        elif self.method == 'bl1':
            self.params = {}
        else:
            raise Exception('unknow method')

    def summary(self):
        summary = {}
        summary['method'] = self.method
        summary['filtration'] = self.filtration
        summary['dataset'] = self.dataset
        summary['params'] = self.params
        return summary

    @staticmethod
    def my_product(inp):
        from itertools import product
        return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

    def n_search(self):
        searchrange = self.params
        return int(np.product(map(len, searchrange.values())))

    def make_hyper_direct(self, make_flag=False, **kwargs):
        # generate the directory for a particular set of hyper-parameters
        # kwargs are one particular set of hyepr parameter

        # make directory for some hyperparameter combination
        # TODO fix direct
        direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/experiments/' \
                 + self.dataset +  '/' + self.method + '/' + self.filtration + '/'

        for parameter in np.sort(self.params.keys()): # sort the param lexigraphically
            if parameter in kwargs.keys():
                direct += parameter + '_' + str(kwargs[parameter]) + '/'

        if make_flag == True: make_direct(direct)
        self.direct = direct

    def comp(self):
        # compare result
        direct = self.direct

    def get_one_hyper(self, idx=1): # need to modify hyper-param
        # from searchrange, get 1 hyperparamater combination
        # searchrange example: {'K': [1], 'bw': [1, 0.1, 10, 0.01, 100, 0.01], 'p': [1]}
        searchrange = self.params
        n = np.product(map(len, searchrange.values()))
        if n == 0:
            result = {}
        else:
            assert (n-1) >= idx
            result =  list(self.my_product(searchrange))[idx]
        if self.method == 'sw':
            result['n_directions'] = 10
        result['method'] = self.method
        result['filtration'] = self.filtration
        result['T/F'] = self.tf

        self.one_param = result
        return result

    def save_one_param(self):
        with open(self.direct + self.suffix + 'param.json', 'w') as fp:
            json.dump(self.one_param, fp, indent=1)

    def load_param(self):
        with open(self.direct + self.suffix + 'eval.json', 'r') as f:
            jdict = json.load(f)

        def pretty_print(jdict):
            pass
        return jdict

    def save_kernel(self, kernel, name='kernel'):
        np.save(self.direct + self.suffix + name, kernel)

    def load_kernel(self, kernel, name='kernel'):
        return np.load(self.direct + self.suffix + name)

    def load_kernel(self):
        try:
            tda_kernel = np.load(self.direct + self.suffix + 'kernel.npy')
            return tda_kernel
        except:
            raise Exception('No precomputed kernel at %s'% (self.direct + self.suffix + 'kernel.npy'))

    def save_best_result(self, eval_result, dgmstat):
        with open(self.direct + self.suffix + 'eval.json', 'w') as fp:
            json.dump({'eval': eval_result, 'dgmstat': dgmstat}, fp, indent=1)

    def load_result(self):
        pass

def dgms_stats(dgms):
    lenlist = map(len, dgms)
    distinct_list = map(n_distinct, dgms)

    return {'min': np.min(lenlist), 'max': np.max(lenlist),
            'ave': np.average(lenlist), 'std': np.std(lenlist),
            'min_distinct': np.min(distinct_list),
            'max_distinct': np.max(distinct_list),
            'ave_distinct': np.average(distinct_list),
            'std_distinct': np.std(distinct_list)}

def load_kernel( graph='mutag', method='sw', normal_flag = True, debug_flag = False, filtration = 'cc', **kwargs):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/experiments/' # \
             # 'mutag/sw/cc/bw_0.1/normal_kernel.npy'
    import os
    direct += os.path.join(graph, method, filtration, '')
    for key, val in kwargs.items():
        direct += os.path.join(key + '_' + str(val), '')
    if debug_flag:
        print (direct)
    if normal_flag == True:
        file = 'normal_kernel.npy'
    else:
        file = 'fake_kernel.npy'
    kernel = np.load(direct + file)
    assert np.max(np.abs((kernel - kernel.T))) < 1e-5
    return kernel
# args_ = {'bw': 0.1}
# load_kernel( graph='mutag', method='sw', normal_flag = True, filtration = 'cc', **args_)

# for idx in range(6):
#     print get_one_hyper(sw_param, idx = idx)
def load_stat(direct, graph, ):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation'

    with open('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/experiments/ptc/sw/deg/bw_10/fake_0_eval.json','r') as f:
        jdict = json.load(f)

def graphassertion(g):
    assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"

def mds_vis(dist_matrix, dgms, colors, n_components = 2, filename = 'embedding'):
    assert np.shape(dist_matrix)[0] == len(colors)
    import matplotlib.pyplot as plt
    dist_matrix = distance_matrix(dgms)
    from sklearn.manifold import MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed')
    pos = mds.fit(dist_matrix).embedding_
    plt.scatter(pos[:, 0], pos[:, 1], c=colors, s=10, lw=0,
                label='True Position')
    plt.savefig(generate_directory() + filename)
    plt.close()

def stat(lis, high_order=False):
    lis = [a for a in lis if a!=0.00123]
    # list = angledgm
    if high_order == True:
        pass
    return np.array([np.min(lis), np.max(lis), np.median(lis), np.mean(lis), np.std(lis)])

def bl0(dgms_to_save_, key='deg'):
    graphs = dgms_to_save_['graphs']
    n = len(graphs)
    blfeat = np.zeros((n, 5))
    for i in range(n):
        fval = nx.get_node_attributes(graphs[i][0], key).values()
        blfeat[i] = stat(fval)
    return blfeat

def bl1(dgms):
    n = len(dgms)
    blfeat = np.zeros((n, 5))
    for i in range(n):
        dgm = dgms[i]  # a list of lists
        cval = []
        for p in dgm:
            cval += p
        assert len(cval) == 2 * len(dgm)
        blfeat[i] = stat(cval)
    blfeat = normalize_(blfeat, axis = 0)
    # print blfeat
    return blfeat


# def bl1(dgms_to_save_, key='deg', type='dgms'):
#     # type can be dgms, sub_dgms, super_dgms, epd_dgms
#     dgms = dgms_to_save_['dgms']
#     n = len(dgms)
#     blfeat = np.zeros((n, 5))
#     for i in range(n):
#         dgm = dgms[1] # a list of lists
#         cval = []
#         for p in dgm:
#             cval += p
#         assert len(cval) == 2 * len(dgm)
#         blfeat[i] = stat(cval)
#     return blfeat



