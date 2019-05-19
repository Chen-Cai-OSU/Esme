import sys
sys.path.append('/home/cai.507/Documents/Utilities/dionysus/build/bindings/python')
import dionysus as d
import numpy as np
# from igraph import *
import random
import time
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from Esme.helper.time import timefunction
# random.seed(42)
NUM_FEATURE = 1000; NORMALIIZATION = 1
np.set_printoptions(precision=4)

# Learning filtration related
class io():
    def load_graph(graph, debug='off'):
        # exptect label to be numpy.ndarry of shape (n,). However protein_data is different so have to handle it differently
        assert type(graph) == str
        import pickle
        import os
        import time
        import numpy as np
        GRAPH_TYPE = graph

        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration'
        make_direct(directory)
        inputFile = directory + '/graph+label'
        if os.path.isfile(inputFile):
            start = time.time()
            print('Loading existing files')
            fd = open(inputFile, 'rb')
            (graphs_, labels_) = pickle.load(fd)
            print(('Loading takes %s' % (time.time() - start)))
            return (graphs_, labels_)

        print('Start Loading from dataset')
        file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + GRAPH_TYPE + ".graph"
        if not os.path.isfile(file):
            file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + GRAPH_TYPE + '.graph'
        f = open(file, 'r')
        data = pickle.load(f)
        graphs = data['graph']
        labels = data['labels']
        if debug == 'on':
            print((graph), end=' ')
            print((type(labels),))
            print((np.shape(labels)))
        if graph == 'protein_data':
            labels = labels.reshape(1113, 1)
            labels = np.array([-1] * 663 + [1] * 450)
        elif graph == ('nci1' or 'nci109'):
            labels = np.sign(labels - 0.5)

        print('Finish Loading graphs')
        outputFile = directory + '/graph+label'
        fw = open(outputFile, 'wb')
        dataset = (graphs, labels)
        pickle.dump(dataset, fw)
        fw.close()
        print('Finish Saving data for future use')
        return (graphs, labels)
    load_graph = staticmethod(load_graph)

class dgm():
    pass
    def __init__(self):
        pass
    def generate(self):
        pass
    def add_dgms(self, dgm2):
        pass

class basepError(Exception):
    pass

class beta_name_not_in_allowed(Exception):
    pass

def load_graph(graph, debug='off', single_graph_flag=True):
        # exptect label to be numpy.ndarry of shape (n,). However protein_data is different so have to handle it differently
        assert type(graph) == str
        import pickle
        import os
        import time
        import numpy as np
        GRAPH_TYPE = graph
        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration'
        make_direct(directory)
        inputFile = directory + '/graph+label'
        if os.path.isfile(inputFile):
            start = time.time()
            print('Loading existing files')
            fd = open(inputFile, 'rb')
            if GRAPH_TYPE == 'reddit_12K':
                file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + 'reddit_12K'+ '.graph'
                f = open(file, 'r')
                data = pickle.load(f)
                graphs_ = data['graph']
                labels_ = data['labels']
            else:
                (graphs_, labels_) = pickle.load(fd)
            print(('Loading takes %s' % (time.time() - start)))

            if graph=='ptc':
                graphs_[151]=graphs_[152]
            return (graphs_, labels_)

        print('Start Loading from dataset')
        file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + GRAPH_TYPE + ".graph"
        if not os.path.isfile(file):
            file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + GRAPH_TYPE + '.graph'
        f = open(file, 'r')
        data = pickle.load(f)
        if single_graph_flag==True:
            import sys
            sys.path.append('/home/cai.507/Documents/DeepLearning/GraphSAGE')

        graphs = data['graph']
        if graph=='ptc':
            graphs[151]=graphs[152]

        labels = data['labels']
        if debug == 'on':
            print((graph), end=' ')
            print((type(labels),))
            print((np.shape(labels)))
        if graph == 'protein_data':
            labels = labels.reshape(1113, 1)
            labels = np.array([-1] * 663 + [1] * 450)
        elif graph == ('nci1' or 'nci109'):
            labels = np.sign(labels - 0.5)

        print('Finish Loading graphs')
        outputFile = directory + '/graph+label'
        fw = open(outputFile, 'wb')
        dataset = (graphs, labels)
        pickle.dump(dataset, fw)
        fw.close()
        print('Finish Saving data for future use')
        return (graphs, labels)

def graph_processing(g, beta = np.array([1,0,0,0])):
    # (g, fv_list) = add_function_value(g, fv='closeness_centrality')

    g = function_basis(g) # belong to pipe0
    g = fv(g, beta)       # belong to pipe1
    (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='max') # belong to pipe1
    # fv_list.sort(reverse=True)
    return g

def convert(dgms_tuple):
    # convert tuple to pd
    import dionysus as d
    return d.Diagram(dgms_tuple)

def fv(g, beta, label='y', hop_flag='n', basep=0, rs=100, edge_fil='off'):
    '''
     beta: [deg, ricci, fiedler, cc, label]
     compute the linear combination(beta is weights) of different filtration function. Store it in fv_test
    '''

    # g = graphs_[0][0]
    # g.node[0].keys()
    # g_after = fv(g, np.array([1,0,0,0,0]), hop_flag='y', basep='c')
    # g_after.node[15]['fv_test']
    import numpy as np
    # assert type(beta) == np.ndarray # relax to accompodate hop_c
    if (edge_fil != 'off'):
        if (edge_fil[-3:]=='ave'):
            for n in g.nodes():
                g.node[n]['fv_test'] = g.node[n][edge_fil]
        else:
            assert ((edge_fil[-3:]=='min') or (edge_fil[-3:]=='max'))
            for n in g.nodes():
                g.node[n]['fv_test'] = g.node[n][edge_fil]
        return g
    if label=='y':
        assert ((len(beta) == 5) or (beta[0:3]=='hop'))
        # assert min(beta) >= 0
        # assert abs(np.sum(beta) -1) < 0.01
        # assert abs(np.linalg.norm(beta)-1) < 0.01 # no normalization
        if hop_flag == 'y':
            for n in g.nodes():
                assert len(g.node[n]['hop']) == len(g)

            if basep == 'c': # center
                import networkx as nx
                closeness_centrality = nx.closeness_centrality(g)
                basep = max(closeness_centrality, key=closeness_centrality.get)
                basep_ = list(g.nodes()).index(basep)
            elif basep == 'r':
                n = len(g)
                rng = random.Random(rs)
                basep_ = rng.randint(0, n-1)
                # basep_ = list(g.nodes())[randidx]
            elif type(int(basep))==int:
                if basep in g.nodes():
                    basep_ = list(g.nodes()).index(basep)
                else:
                    basep_ = 'dummy'
            else:
                print(basep)
                raise basepError('Basepoint Error. Unconsidered case occured')

            for v in g.nodes():
                assert 'hop' in list(g.node[v].keys())
            for n in g.node():
                if basep_ == 'dummy': # expected case
                    g.node[n]['fv_test'] = -10**(-5)
                else:
                    try:
                        g.node[n]['fv_test'] = g.node[n]['hop'][basep_]
                    except: # true expection
                        print(('Basep is %s, Basep_ is %s'%(basep, basep_)))
                        print(('Number of nodes is', len(g)))
                        print(('Hop feature dim is ', np.shape(g.node[n]['hop'])))
                        g.node[n]['fv_test'] = 3e-5
            return g

        # add a shortcut to save some time
        if (beta == np.array([1,0,0,0,0])).all():
            for n in g.nodes():
                g.node[n]['fv_test'] = g.node[n]['deg']
            return g

        if (beta == np.array([0,1,0,0,0])).all():
            for n in g.nodes():
                g.node[n]['fv_test'] = g.node[n]['ricciCurvature']
            return g

        if (beta == np.array([0,0,1,0,0])).all():
            for n in g.nodes():
                g.node[n]['fv_test'] = g.node[n]['fiedler']
            return g

        if (beta == np.array([0,0,0,1,0])).all():
            for n in g.nodes():
                g.node[n]['fv_test'] = g.node[n]['cc']
            return g

        if (beta == np.array([0, 0, 0, 0, 1])).all():
            for n in g.nodes():
                g.node[n]['fv_test'] = g.node[n]['label']
            return g

        assert 'deg' and 'ricciCurvature' and 'fiedler' and 'cc' and 'label' in list(g.node[list(g.nodes())[0]].keys())
        for n in g.nodes():
            g.node[n]['fv_test'] = beta[0] * g.node[n]['deg'] + beta[1] * g.node[n]['ricciCurvature'] \
                                   + beta[2] * g.node[n]['fiedler'] + beta[3] * g.node[n]['cc'] + beta[4] * g.node[n]['label']
    elif label=='n':
        assert len(beta) == 4
        # assert min(beta) >= 0
        assert abs(np.sum(beta) - 1) < 0.01
        assert 'deg' and 'ricciCurvature' and 'fiedler' and 'cc' in list(g.node[list(g.nodes())[0]].keys())  # didn't add label
        for n in g.nodes():
            g.node[n]['fv_test'] = beta[0] * g.node[n]['deg'] + beta[1] * g.node[n]['ricciCurvature'] \
                                   + beta[2] * g.node[n]['fiedler'] + beta[3] * g.node[n]['cc']
    return g

def format_matrixprl(matrix_prl):
    # format work. Input matrix_prl. Outpt (dist_matrix, idx_dict)
    n = np.shape(matrix_prl)[0]
    dist_matrix = np.zeros((n, n))
    idx_dict = {}
    for i in range(n):
        dist_matrix[i] = matrix_prl[i][0]
        idx_dict.update(matrix_prl[i][1])
    for i in range(n):
        for j in range(i):
            idx_dict[(i, j)] = idx_dict[(j, i)][::-1] # reverse a tuple

        # idx_dict = merge_two_dicts(idx_dict, matrix_prl[i][1])
    return ((dist_matrix + dist_matrix.T), idx_dict)

def add_dgms(dgm1, dgm2):
    def convert(dgm):
        # convert dgm to a list of tuples(length 2)
        return [(np.float(p.birth), np.float(p.death)) for p in dgm]
    data = convert(dgm1) + convert(dgm2)
    # print(data)
    return d.Diagram(data)

def set_betalist(allowed):
    if check_edge_filtration(allowed):
        betalist = [unit_vector(5, 1)]  # dummy
    else:
        betalist = [ unit_vector(5,0), unit_vector(5,1), unit_vector(5,3)] # remove 'hop_c' for regression test # also remove fiedler and label
    return betalist

def sklearn_tda():
    import sklearn_tda as tda
    import matplotlib.pyplot as plt
    import numpy as np

    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))

    D = np.array([[0.0, 4.0], [1.0, 2.0], [3.0, 8.0], [6.0, 8.0]])
    plt.scatter(D[:, 0], D[:, 1])
    plt.plot([0.0, 10.0], [0.0, 10.0])
    plt.show()

    diags = [D]

    LS = tda.Landscape(resolution=1000)
    L = LS.fit_transform(diags)
    plt.plot(L[0][:1000])
    plt.plot(L[0][1000:2000])
    plt.plot(L[0][2000:3000])
    plt.show()

    SH = tda.Silhouette(resolution=1000, weight=lambda x: np.power(x[1] - x[0], 5))
    S = SH.fit_transform(diags)
    plt.plot(S[0])
    plt.show()

    BC = tda.BettiCurve(resolution=1000)
    B = BC.fit_transform(diags)
    plt.plot(B[0])
    plt.show()

    diagsT = tda.DiagramPreprocessor(use=True, scaler=tda.BirthPersistenceTransform()).fit_transform(diags)
    PI = tda.PersistenceImage(bandwidth=1.0, weight=arctan(1.0, 1.0), im_range=[0, 10, 0, 10], resolution=[100, 100])
    I = PI.fit_transform(diagsT)
    plt.imshow(np.flip(np.reshape(I[0], [100, 100]), 0))
    plt.show()

    plt.scatter(D[:, 0], D[:, 1])
    D = np.array([[1.0, 5.0], [3.0, 6.0], [2.0, 7.0]])
    plt.scatter(D[:, 0], D[:, 1])
    plt.plot([0.0, 10.0], [0.0, 10.0])
    plt.show()

    diags2 = [D]

    SW = tda.SlicedWassersteinKernel(num_directions=10, bandwidth=1.0)
    X = SW.fit(diags)
    Y = SW.transform(diags2)
    print(("SW  kernel is " + str(Y[0][0])))

    PWG = tda.PersistenceWeightedGaussianKernel(bandwidth=1.0, weight=arctan(1.0, 1.0))
    X = PWG.fit(diags)
    Y = PWG.transform(diags2)
    print(("PWG kernel is " + str(Y[0][0])))

    PSS = tda.PersistenceScaleSpaceKernel(bandwidth=1.0)
    X = PSS.fit(diags)
    Y = PSS.transform(diags2)
    print(("PSS kernel is " + str(Y[0][0])))

    W = tda.WassersteinDistance(wasserstein=1, delta=0.001)
    X = W.fit(diags)
    Y = W.transform(diags2)
    print(("Wasserstein-1 distance is " + str(Y[0][0])))

    sW = tda.SlicedWassersteinDistance(num_directions=10)
    X = sW.fit(diags)
    Y = sW.transform(diags2)
    print(("sliced Wasserstein distance is " + str(Y[0][0])))

def dgm_vec(diags, vec_type='pi', axis=1):
    import time
    t1 = time.time()
    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))
    import sklearn_tda as tda
    if vec_type=='pi':
        diagsT = tda.DiagramPreprocessor(use=True, scaler=tda.BirthPersistenceTransform()).fit_transform(diags)
        PI = tda.PersistenceImage(bandwidth=1.0, weight=arctan(1.0, 1.0), im_range=[0, 1, 0, 1], resolution=[25, 25])
        I = PI.fit_transform(diagsT)
        res = I

    elif vec_type == 'pl':
        LS = tda.Landscape(num_landscapes=5, resolution=100)
        L = LS.fit_transform(diags)
        res = L
    t2 = time.time()
    t = precision_format((t2 - t1),1)
    return (remove_zero_col(normalize_(res, axis=axis)), t)

def kernel_parameter( kernel_type='sw'):
    if kernel_type=='sw':
        # bw = [0.01, 0.1, 1, 10, 100]
        bw = [1, 0.1, 10, 0.01, 100, 0.01]
        k = [1]; p=[1];
    elif kernel_type == 'pss':
        # bw = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100, 500, 1000]
        bw = [1, 5e-1, 5, 1e-1, 10, 5e-2, 50, 1e-2, 100, 5e-3, 500, 1e-3, 1000]
        k = [1]; p = [1];
    elif kernel_type == 'wg':
        bw = [1, 10, 0.1, 100, 0.01]
        # k = [1];
        # p = [1];
        k = [ 1, 10, 0.1];
        p = [ 1, 10, 0.1];
    return {'bw':bw, 'K':k, 'p':p}

def sw(dgms1, dgms2, parallel_flag=False, kernel_type='sw', n_directions=10, bandwidth=1.0, K=1, p = 1):
    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))

    import sklearn_tda as tda
    if parallel_flag==False:
        if kernel_type=='sw':
            tda_kernel = tda.SlicedWassersteinKernel(num_directions=n_directions, bandwidth=bandwidth)
        elif kernel_type=='pss':
            tda_kernel = tda.PersistenceScaleSpaceKernel(bandwidth=bandwidth)
        elif kernel_type == 'wg':
            tda_kernel = tda.PersistenceWeightedGaussianKernel(bandwidth=bandwidth, weight=arctan(K, p))
        else:
            print ('Unknown kernel')

        diags = dgms1; diags2 = dgms2
        X = tda_kernel.fit(diags)
        Y = tda_kernel.transform(diags2)
        return Y

def assert_sw_dgm(dgms):
    # check sw_dgm is a list array
    # assert_sw_dgm(generate_swdgm(10))
    assert type(dgms)==list
    for dgm in dgms:
        assert np.shape(dgm)[1]==2

def sw_parallel(dgms1, dgms2, parallel_flag=True, kernel_type='sw', n_directions=10, bandwidth=1.0, K = 1, p = 1, granularity=25, **kwargs):
    import time
    t1 = time.time()
    assert_sw_dgm(dgms1)
    assert_sw_dgm(dgms2)
    from joblib import Parallel, delayed
    n1 = len(dgms1); n2 = len(dgms2)
    kernel = np.zeros((n2, n1))

    if parallel_flag==False:         # used as verification
        for i in range(n2):
            kernel[i] = sw(dgms1, [dgms2[i]], kernel_type=kernel_type, n_directions=n_directions, bandwidth=bandwidth)
    if parallel_flag==True:
        # parallel version
        kernel = Parallel(n_jobs=-1)(delayed(sw)(dgms1, dgms2[i:min(i+granularity, n2)], kernel_type=kernel_type, n_directions=n_directions, bandwidth=bandwidth, K=K, p=p) for i in range(0, n2, granularity))
        kernel = (np.vstack(kernel))
    return (kernel/float(np.max(kernel)), precision_format(time.time()-t1, 1))

def generate_swdgm(size=100):
    import numpy as np
    dgm = []
    for i in range(size):
        dgm += [np.random.rand(100,2)]
    return dgm

def assert_dgm_below(dgm):
    i = 0
    for p in dgm:
        if p.birth < p.death:
            raise Exception('Dgm above diagonal. Found %s point, birth is %s, and death is %s'%(i, p.birth,p.death))
        i = i + 1
def assert_dgm_above(dgm):
    i = 0
    for p in dgm:
        if p.birth > p.death:
            raise Exception('Dgm below diagonal. Found %s point, birth is %s, and death is %s'%(i, p.birth,p.death))
        i = i + 1

def flip_dgm(dgm):
    # flip dgm from below to above, not vise versa
    import dionysus as d
    for p in dgm:
        if np.float(p.birth) < np.float(p.death):
            assert_dgm_above(dgm)
            return dgm
        assert np.float(p.birth) >= np.float(p.death)
    data = [(np.float(p.death), np.float(p.birth)) for p in dgm]
    return d.Diagram(data)

def normalize_dgm(dgm):
    import numpy as np
    max_ = 0
    for p in dgm:
        max_ = max(max_, max(np.float(abs(p.birth)), np.float(abs(p.death))))
    max_ = np.float(max_)
    data = [(np.float(p.birth)/max_, np.float(p.death)/max_) for p in dgm]
    return d.Diagram(data)

# print_dgm(flip_dgm(super_dgms[0]))
def equal_dgm(dgm1, dgm2):
    import dionysus as d
    if d.bottleneck_distance(dgm1, dgm2)!=0:
        return True
    else:
        return False

def generate_graphs(version=2):
    # Return the list of all graphs with up to seven nodes named in the Graph Atlas.
    import networkx as nx
    if version==1:
        graphs = []
        n = len(nx.graph_atlas_g())
        for i in range(n):
            graphs.append([nx.graph_atlas(i)])
        return graphs
    elif version==2:
        graphs = nx.graph_atlas_g()
        graphs = [[g] for g in graphs]
        return graphs[1:]

def read_all_graphs(n=5):
    # get all non-isomorphic graphs of node n
    path = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/nauty/'
    import networkx as nx
    graphs = nx.read_graph6(path + 'graph' + str(n) + 'c.g6')
    graphs = [[graph] for graph in graphs]
    return graphs
def filtration_repr(beta):
    # input: numpy.array like [1,1,0,0,0]
    # ouput a string deg(0.5) + ricci(0.5)
    beta = beta / float(sum(beta))
    beta_dict = {0: 'deg', 1: 'ricci', 2: 'fiedler', 3: 'cc', 4:'label'}
    output = ''
    for i in range(5):
        if beta[i] > 1e-5:
            output += beta_dict[i] + '(' + str(beta[i]) + ') + '
    return output
@timefunction
def prob_prefiltration(G, H=0.3, sub_flag=True):
    """
    # take a netowrkx graph and compute the edge probability for each edge and return the same graph
    """
    import sys
    import numpy as np
    import networkx as nx
    sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/')
    import graphonlib
    # G = getG();
    assert nx.is_connected(G)
    print(summary(G))
    Ga = np.array(nx.to_numpy_matrix(G))
    try:
        Gp = np.array(graphonlib.smoothing.zhang.smoother(Ga,h=H)) # h is a parameter
        assert (Gp == Gp.T).all()
    except:
        print('Graphon Computation Error')
        Gp = np.zeros(np.shape(Ga))
    p_matrix = np.multiply(Ga, Gp); assert (p_matrix == p_matrix.T).all()

    for v in G.nodes():
        v_nbr = list(nx.neighbors(G, v))
        v_nbr_ = [list(G.nodes()).index(n) for n in v_nbr]
        # print (v, v_nbr, v_nbr_)

        v_ = list(G.nodes()).index(v)
        if sub_flag==True:
            tmp = [p_matrix[v_][idx] for idx in v_nbr_ ]
            G.node[v]['edge_p'] = min(tmp)

        if sub_flag == False:
            G.node[v]['edge_p'] = max(p_matrix[v_])

    node_idx = np.max(G.nodes()) + 1
    old_edges = list(G.edges()) # avoid inplace change
    for s,t in old_edges:
        s_ = list(G.nodes()).index(s);  t_ = list(G.nodes()).index(t)
        G.add_node(node_idx, edge_p = p_matrix[s_][t_])
        G.add_edges_from([(s, node_idx), (node_idx, t)])
        try:
            G.remove_edge(s, t)
        except: pass
        node_idx +=1
    return G

class test_dionysus():
    pass

    def test_dionysus(self):
        import dionysus as d
        dgm = d.Diagram([(1, 2), (3, 4)])
        for i in range(100):
            print((dgm[i]))
        for p in dgm:
            print(p)

    def computePD(i):
        # print('Process %s' % i)
        import dionysus as d
        import numpy as np
        # np.random.seed(42)
        f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
        m1 = d.homology_persistence(f1)
        dgms1 = d.init_diagrams(m1, f1)
        # return [(p.birth, p.death) for p in dgms1[1]]  # advice from Dmitriy
        return dgms1[1]
    computePD = staticmethod(computePD)

    def bad_example():
        import dionysus as d
        dgm1 = d.Diagram([(1, 2.07464)])
        dgm1 = d.Diagram([(1, 2.04287)])
        dgm2 = d.Diagram([(1, 1.68001), (1, 1.68001), (1, 1.68001)])  # this one doesn't work
        dgm2 = d.Diagram([(1, 1.71035)])
        # dgm2 = d.Diagram([(1,1.68), (1,1.68), (1,1.68)]) # But this one works
        print((d.bottleneck_distance(dgm1, dgm2)))
        print((d.bottleneck_distance_with_edge(dgm1, dgm2)))

    def computePD(i):
        # print('Process %s' % i)
        import dionysus as d
        import numpy as np
        # np.random.seed(42)
        f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
        m1 = d.homology_persistence(f1)
        dgms1 = d.init_diagrams(m1, f1)
        # return [(p.birth, p.death) for p in dgms1[1]]  # advice from Dmitriy
        return dgms1[1]

    def test_get_dgms(n_jobs=1):
        from joblib import delayed, Parallel
        return Parallel(n_jobs=-1)(delayed(computePD)(i) for i in range(70))


# get the persistence vector
def gaussian(x, mu, sig):
    return np.exp(-np.power(x-mu,2)/(2 * np.power(sig,2) + 0.000001))

def data_interface(dgm, dynamic_range_flag=True):
    data = [tuple(i) for i in dgm2diag(dgm)]
    try:
        [list1, list2] = list(zip(*data));
    except:
        list1 = [0]; list2 =[1e-5] # adds a dummy 0
    if dynamic_range_flag == True:
        min_ = min(min(list1), min(list2))
        max_ = max(max(list1), max(list2))
        std_ = (np.std(list1) + np.std(list2))/2.0
    elif dynamic_range_flag == False:
        min_ = -5
        max_ = 5
        std_ = 3

    return {'data': data, 'max': max_ + std_, 'min': min_ - std_}

def rotate_data(data,super_check):
    """
    :param data:
    :return: a list of tuples
    """
    def rotate(x, y):
        return np.sqrt(2) / 2 * np.array([x + y, -x + y])
    def flip(x,y):
        return np.array([y,x])

    length = len(data)
    rotated = []; point = [0,0];
    for i in range(0,length,1):
        if super_check == True:
            data[i] = flip(data[i][0],data[i][1])
        point = rotate(data[i][0], data[i][1]);
        point = (point[0], point[1])
        rotated.append(point)
    return rotated

def searchclf(i):
    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=i)
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=1)
        clf.fit(X_train, y_train)
        print(("Best parameters set found on development set is \n %s with score %s" % (
        clf.best_params_, clf.best_score_)))
        print((clf.best_params_))
        print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print(("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params)))
        print("Detailed classification report:\n")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print((classification_report(y_true, y_pred)))

def evaluate_clf__(X, Y):
    from sklearn.model_selection import StratifiedKFold
    from sklearn import svm
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + 'neuron' + '/result/'
    accuracy = []
    n = 10
    for i in range(n):
        # after grid search, the best parameter is {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
        clf = svm.SVC(kernel='rbf', gamma=0.1, C=400)
        # clf = svm.SVC(C=40, kernel='linear')
        from sklearn.model_selection import cross_val_score
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        cvs = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold)
        print(cvs)
        acc = cvs.mean()
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    print(('mean is %s, std is %s '%(accuracy.mean(), accuracy.std())))

def persistence_repr(dgms, dynamic_range_flag=True):
    from joblib import Parallel, delayed
    X = Parallel(n_jobs=-1)(delayed(persistence_vector)(dgms[i], dynamic_range_flag=dynamic_range_flag) for i in range(len(dgms)))
    n = len(dgms)
    data = np.zeros((n, 500))
    for i in range(n):
        data[i] = X[i]
    from sklearn.preprocessing import normalize
    data = normalize(data)
    return data


def evaluate_best_estimator(grid_search_re, X, Y, print_flag='off'):
    if print_flag=='on': print('Start evaluating the best estimator')

    # start multiple cv
    param = grid_search_re['param']
    assert (param['kernel'] == 'linear') or (param['kernel'] == 'rbf')
    assert isinstance(param, dict)
    start = time.time()
    if len(param) == 3:
        clf = svm.SVC(kernel='rbf', C=param['C'], gamma = param['gamma'])
    elif (len(param) == 2) and (param['kernel'] == 'linear'):
        clf = svm.SVC(kernel='linear', C = param['C'])
    else:
        print('Unconsidered cases in evaluate_best_estimator')

    # the easy cv method
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    # print('this is easy cv')
    now = time.time()
    cv_score = []
    n_cv = 5
    for i in range(n_cv):
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        if param['kernel']!= 'precomputed':
            scores = cross_val_score(clf, X, Y, cv=k_fold, scoring='accuracy', n_jobs=-1)

        # should be bug
        elif param['kernel']!= 'precomputed':
            scores = cross_val_score(clf, lap, Y, cv=k_fold, scoring='accuracy', n_jobs=-1)
        assert param['kernel']!= 'precomputed'

        if print_flag == 'on': print(scores)
        cv_score.append(scores.mean())

    cv_score = np.array(cv_score)
    end = time.time()

    print(('Evaluation takes %0.3f. '
          'After averageing %0.1f cross validations, the mean accuracy is %0.3f, the std is %0.3f\n'
          %(end-now, n_cv, cv_score.mean(), cv_score.std())))
    return (cv_score.mean(), cv_score.std())



# Auxliary function
def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        # print(pt),
        # print(type(pt))
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag

def dgms2diags(dgms):
    diags = []
    for i in range(len(dgms)):
        diags.append(dgm2diag(dgms[i]))
    return diags


def dgm2diag_(dgm):
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append((pt.birth, float('Inf')))
        else:
            diag.append((pt.birth, pt.death))
    # print(diag)
    return diag
def dgm2array(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    import numpy as np
    return np.array(dgm2diag(dgm))
def d_bottleneck_distance_with_edge_base(diag1, diag2, bd=0.75, debug='off'):
    # only works for bottleneck distance
    # diag1, diag2 are np.array

    def proj(pt):
        assert type(pt) == list
        assert len(pt) == 2
        coordinate = (pt[0] + pt[1]) * 0.5
        if coordinate == float('inf'):
            coordinate = 0
        return [coordinate, coordinate]
    def musk(n1, n2):
        # generate a musk
        # to make the bottom right submatrix big
        n = n1 + n2
        background = np.zeros((n, n))
        musk = np.zeros((n1, n2)) + 1e4
        background[n2:n, n1:n] = background[n2:n, n1:n] + musk
        return background

    import numpy as np
    from scipy.spatial.distance import cdist
    n1 = len(diag1); n2 = len(diag2)
    assert np.shape(diag1)[1] == 2
    assert np.shape(diag2)[1] == 2

    # diag1_tmp = copy.deepcopy(diag1)
    # for n in diag2:
    #     diag1.append(proj(n)) # will have side effect
    # assert len(diag1) == n1+n2
    # for n in diag1_tmp:
    #     diag2.append(proj(n)) # will have side effect
    # assert len(diag2) == len(diag1) == n1+n2

    # print('diag1 is %s'%diag1)
    # print('diag2 is %s' % diag2)
    dist_matrix = np.zeros((2*n1+n2, n2)) + 1e5
    dist_matrix[0:n1, 0:n2] = cdist(diag1, diag2, 'chebyshev')
    assert np.shape(dist_matrix) == (2* n1+n2, n2)
    # dist_matrix = dist_matrix + musk(n2, n1)
    for i in range(n1, 2*n1):
        dist_matrix[i, 0] = abs(diag1[i-n1][1] - diag1[i-n1][0])/2.0
    for i in range(2*n1, 2*n1+n2):
        dist_matrix[i, 0] = abs(diag2[i-2*n1][1] - diag2[i-2*n1][0])/2.0

    difference = np.nanmin(abs(dist_matrix - bd))
    if debug == 'on':
        print((abs(dist_matrix - bd)))
    assert difference < 0.02
    idx = np.nanargmin((abs(dist_matrix - bd)))
    row_idx = idx / np.shape(dist_matrix)[1]
    col_idx = idx % np.shape(dist_matrix)[1]
    assert np.shape(dist_matrix)[1] == np.shape(diag2)[0]
    if (row_idx < n1) and (col_idx<=n2):
        p1 = diag1[row_idx]; p2 = diag2[col_idx]
        if debug=='on':
            print(('P1 is %s' % p1), end=' ')
            print(('P2 is %s' % p2), end=' ')
            print(('BD is %s' % bd))
            err = (cdist([p1], [p2], 'chebyshev') - bd)[0][0]
            print(('Error is %s'%err))
        assert (cdist([p1], [p2], 'chebyshev') - bd)[0][0] < 0.02
        try:
            assert (cdist([p1], [p2], 'chebyshev') - bd)[0][0] < 0.02
        except AssertionError:
            print(('P1 is %s'%p1), end=' ')
            print(('P2 is %s'%p2), end=' ')
            print(('BD is %s'%bd))
            sys.exit()
    elif (n1<=row_idx<2*n1) and (col_idx==0):
        p1 = diag1[row_idx-n1]; p2 = [0,0] # p2 is dummy here
        assert (abs(p1[0] - p1[1])/2.0 -bd) < 0.01
    elif (2*n1 <= row_idx < 2*n1+n2) and (col_idx == 0):
        p2 = diag2[row_idx - 2*n1]; p1 = [0, 0]  # p1 is dummy here
        assert (abs(p2[0] - p2[1]) / 2.0 - bd) < 0.01
        # try:
        #     assert (abs(p2[0] - p2[1])/2.0 - bd) < 0.01
        # except AssertionError:
        #     print('Assertion fails, p2 is %s, %s'%(p2[0], p2[1])),
        #     print('BD is %s'%bd)
    def same_pts(p1, p2):
        # check if two points are approximately same.
        # p1: [x, y]. p2:[x, y]
        assert len(p1) == len(p2) == 2
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))<0.01
    # same_pts([1,2], [1.001,2])

    def get_indx(p, dgm, debug='off'):
        # get the index of p in dgm. p is of form [x, y]
        # if p is on diagonal, output -1
        if p[0]==p[1]:
            return -1
        i = 0
        for pts in dgm:
            # if p == [pts.birth, pts.death]:
            if same_pts(p, [pts.birth, pts.death]):
                return i
            else:
                i = i +1
        return 'No pt found match'
    # dgm_test = d.Diagram([(1,2), (2,3)])
    # dgm_test = diag2dgm(diag2)
    # get_indx([6.1, 20], dgm_test)
    idx1 = get_indx(p1, diag2dgm(diag1))
    idx2 = get_indx(p2, diag2dgm(diag2))
    if debug == 'on':
        print((idx1), end=' ')
        print (idx2)
    return (bd, (idx1, idx2))
    # return {'p1': diag1[row_idx], 'p2': diag2[col_idx]}
def d_bottleneck_distance_with_edge(dgm1, dgm2, bd=0.75, debug='off'):
    assert str(type(dgm1)) == "<class 'dionysus._dionysus.Diagram'>"
    assert str(type(dgm2)) == "<class 'dionysus._dionysus.Diagram'>"

    diag1 = dgm2array(dgm1)
    diag2 = dgm2array(dgm2)
    return d_bottleneck_distance_with_edge_base(diag1, diag2, bd, debug=debug)
def print_stoa(table, graph, n_row):
    for method_ in list(stoa().keys()):
        graph_local = graph
        if graph=='dd_test':
            graph_local = 'dd'
        try:
            accuracy = stoa()[method_][graph_local]
        except:
            accuracy = 'N/A'
        table.add_row([method_, accuracy] + ['']*(n_row-2))
    table.add_row([' ']*n_row)
    print(table)
    return table




def read_multibasept_pd(graph, i):
    import dionysus as d
    file = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/prefiltration/d_multi_basepoints/' + \
        str(i) + '_ss.txt'
    f = open(file, 'r')
    contents = f.read()
    contents = contents.split('\n')
    contents = [p.split(' ') for p in contents][:-1]
    contents = [(float(p[0]), float(p[1])) for p in contents if p[0]>=0]
    # contents = [p for p in contents if p[0]> 1e-2]
    dgm = d.Diagram(contents)
    return dgm

def read_all_basepd(graph, i):
    # graph = 'mutag'; i = 183
    import numpy as np
    file = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/prefiltration/d_all_multi_basepoints/' + \
           str(i) + '_ss.txt'
    f = open(file, 'r')
    contents = f.read()
    contents = contents.split('\n')
    contents = [p.split(' ') for p in contents][:-1]
    contents = [(float(p[0]), float(p[1])) for p in contents]
    contents = [(np.floor(p[0]), np.floor(p[1])) for p in contents if p[0]>=0] # rounding to remove 1e-5
    return contents # a list of tuples

def pdpoint_hist(data, upper_bound=20):
    from itertools import combinations
    upbd = upper_bound
    points = list(combinations(list(range(upbd)),2))
    vector = np.zeros(len(points))
    s = 0
    for pt in data:
        try:
            assert pt in points
        except AssertionError:
            print(pt)
    for p in points:
        # print(p),
        # print(data.count(p))
        vector[s] = data.count(p)
        s +=1
    try:
        assert np.sum(vector)==len(data)
    except AssertionError:
        print(('vector sum is %s'%np.sum(vector)), end=' ')
        print(('data length is %s'%len(data)))
    return vector

def basept_individual_vector(graph, i, upper_bound=20):
    data = read_all_basepd(graph, i)
    return pdpoint_hist(data, upper_bound)
def normalize_(X, axis=0):
    from sklearn.preprocessing import normalize
    return normalize(X, axis=axis)

def basept_vector(graph, n, normalize_ax=1, upper_bound=20):
    from joblib import delayed, Parallel
    # graph = 'mutag'; n = 100
    X = Parallel(n_jobs=-1)(delayed(basept_individual_vector)(graph, i, upper_bound) for i in range(n))  # compute matrix in parallel
    X = np.vstack(X)
    assert np.shape(X)[0]==n
    return normalize_(X, axis=normalize_ax)

def graphs_summary(graphs_):
    for gs in graphs_:
        n_nodes = np.sum([len(g) for g in gs])
        n_edges = np.sum([len(g.edges()) for g in gs])
        print(('Nodes:', n_nodes, 'Edges:', n_edges))

def dist_distribution(g, cdf_flag=False):
        import networkx as nx
        # g = graphs_[0][0]
        if g== None:
            return np.zeros((1,30))
        assert nx.is_connected(g)
        dist_dict = dict(nx.all_pairs_shortest_path_length(g))
        dist_distribution_tmp = [list(dist_dict[i].values()) for i in list(dist_dict.keys())]
        dist_distribution_ = [val for sublist in dist_distribution_tmp for val in sublist]
        # for v1 in g.nodes():
        #     for v2 in g.nodes():
        #         dist_distrbution += [dist_dict[v1][v2]]

        assert len(dist_distribution_) == len(g)**2
        if cdf_flag==True:
            from statsmodels.distributions.empirical_distribution import ECDF
            ecdf = ECDF(dist_distribution_)
            return ecdf(list(range(0,30)))
        return np.histogram(dist_distribution_, list(range(31)))[0]

@timefunction
def dgms_data_(graph, n):
    dgms = [d.Diagram()] * n
    for i in range(n):
        dgms[i] = read_multibasept_pd(graph, i)
    return dgms

# dgms = dgms_data_('mutag',10)
# np.shape(persistence_vectors(dgms))


def access_global():
    # global a
    print(('globals inside function are ', list(globals().keys())))
    print(a)

@timefunction
def dump_data(GRAPH_TYPE, dataset, dataname, beta=-1, still_dump='yes', skip='yes'):
    if skip=='yes':
        return
    # save dataset in a directory for future use
    import pickle, os
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/Baseline/'; make_direct(directory)
    if (str(type(beta))=="<type 'numpy.ndarray'>") and (len(beta)==5):
        print('Saving in beta subdirectory')
        beta_str = str(beta[0]) + '_' +  str(beta[1]) + '_' + str(beta[2]) + '_' + str(beta[3]) + '_' + str(beta[4])
        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/Baseline/' + beta_str + '/'
        make_direct(directory)

    outputFile = directory + dataname
    if os.path.exists(outputFile):
        if still_dump == 'no':
            print('File already exists. No need to dump again.')
            return
    print('Dumping')
    fw = open(outputFile, 'wb')
    pickle.dump(dataset, fw)
    fw.close()
    print(('Finish Saving data %s for future use'%dataname))

def load_ppi(graph):
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/LearningFiltration/'
    inputFile = directory + 'ppi.graph'
    import pickle
    fd = open(inputFile, 'rb')
    dataset = pickle.load(fd)
    return dataset

def test_dictio():
    a = {'hello': 'world'}
    import pickle
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/cai.507/Documents/DeepLearning/deep-persistence/ppi/LearningFiltration/ppi_data', 'rb') as handle:
        b = pickle.load(handle)
    print((a == b))

# def laplacian_kernel(X, sigma):
#     from scipy.spatial.distance import pdist,squareform
#     dist_matrix = squareform(pdist(X,'euclidean'))
#     return np.exp(dist_matrix/sigma)

def load_data(GRAPH_TYPE, dataname, beta=-1, no_load='yes'):
    # load data from a directory
    import time
    start = time.time()
    if no_load == 'yes':
        return (None, 'Failure')
    import os
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/Baseline/'
    if (str(type(beta))=="<type 'numpy.ndarray'>") and (len(beta)==5):
        beta_str = str(beta[0]) + '_' + str(beta[1]) + '_' + str(beta[2]) + '_' + str(beta[3]) + '_' + str(beta[4])
        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration/' + beta_str + '/'
        make_direct(directory)

    inputFile = directory + dataname
    if os.path.isfile(inputFile):
        import pickle as pickle
        print(('Loading existing files %s'%dataname))
        fd = open(inputFile, 'rb')
        dataset = pickle.load(fd)
        print(('Loading %s takes %s'%(dataname, time.time()- start)))
        return (dataset, 'success')
    else:
        return (None, 'Failure')
def print_best(best):
            for score in best:
                (c, sigma, train_acc, test_acc, test_std) = score[0], score[1], score[2], score[3], score[4]
                print(('C: %s, sigma: %s, train_acc:%s test_acc:%s test_std:%s '%(c, sigma, train_acc, test_acc, test_std)))
def baseline(graph):
    if graph=='mutag':
        return '83/87'
    if graph == 'ptc':
        return '54/63'
    if graph == 'protein_data':
        return '67/76'
    if graph == 'nci1':
        return '65/84'
    if graph == 'nci109':
        return '64/85'
    else:
        return 'NoBaseline'
def make_direct(direct):
    # has side effect
    import os
    if not os.path.exists(direct):
            os.makedirs(direct)
def kernel_summary(K):
    print(('Min is %s Max is %s Average is %s'%( np.min(K), np.max(K), np.average(K))))
# def convert2igraph(graph):
#     from igraph import *
#     keys = graph.keys()
#     try:
#         assert keys == range(len(graph.keys()))
#     except AssertionError:
#         print(keys, len(graph.keys()))
#     gi = Graph()
#     # add nodes
#     for i in keys:
#         gi.add_vertices(1)
#     # add edges
#     for i in keys:
#         for j in graph[i]['neighbors']:
#             if j>i:
#                 gi.add_edges([(i,j)])
#     # add labels
#     for i in keys:
#         gi.vs[i]['id'] = 'Nonesense'
#         gi.vs[i]['label'] = graph[i]['label'][0]
#     # print(gi)
#     # print (gi.vs[1])
#     assert len(gi.vs) == len(graph.keys())
#     return gi
def load_graph_igraph(graph):
    assert type(graph)==str
    import os
    GRAPH_TYPE = graph
    pass
    import pickle
    file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + GRAPH_TYPE + ".graph"
    if not os.path.isfile(file):
        file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + GRAPH_TYPE + '.graph'
    f = open(file, 'r')
    data = pickle.load(f)
    data = data['graph']
    return data
def Kernel2Dist(K):
    import numpy as np
    # K = K / float(np.max(K))
    n = np.shape(K)[0]
    diag = np.diag(K).reshape(n,1)
    k1 = np.concatenate((diag, np.ones((n,1))), axis=1)
    k2 = np.concatenate((np.ones((1,n)), diag.reshape(1,n) ), axis=0)
    print((np.shape(k2)))
    dist_matrix = np.dot(k1, k2)
    dist_matrix = dist_matrix - 2 * K
    print(('Number of zeros before truncation %s'%np.sum(dist_matrix == 0)))
    dist_matrix = dist_matrix * (dist_matrix>0)
    print(('Number of zeros before truncation %s'%np.sum(dist_matrix == 0)))
    return dist_matrix
# @timefunction
def rfclf(X,Y, m_f = 40, multi_cv_flag=False):
    import time
    import numpy as np
    if np.shape(X)==(1,2):
        return
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    score1 = np.array([0, 0])
    score1 = cross_val_score(clf1, X, Y, cv=10, n_jobs=-1)
    ten_score = []
    if multi_cv_flag==True:
        n_iter = 10
        print ('Using rf 10 cv 10 times')
    else:
        n_iter=1
    time1 = time.time()
    for seed in range(n_iter):
        clf2 = RandomForestClassifier(n_estimators=40, random_state = seed, max_features=m_f, max_depth=None, min_samples_split=2,n_jobs=-1)
        score2 = cross_val_score(clf2, X, Y, cv=10, n_jobs=-1)
        ten_score += list(score2)
    score2 = np.array(ten_score)
    time2 = time.time()
    rf_time = precision_format(time2-time1,1)
    print(('Try Random Forest, n_estimators=40, max_features=', m_f), end=' ')
    print(('Accuracy is %s'%score2.mean()))
    return (score1.mean(), score2.mean(), rf_time/100.0)

def unzip_databundle(databundle):
    assert np.shape(databundle)[1] == 5
    graphs = [databundle[i][0] for i in range(len(databundle))]
    dgms = [databundle[i][1] for i in range(len(databundle))]
    sub_dgms = [databundle[i][2] for i in range(len(databundle))]
    super_dgms = [databundle[i][3] for i in range(len(databundle))]
    epd_dgms = [databundle[i][4] for i in range(len(databundle))]
    return (graphs, dgms, sub_dgms, super_dgms, epd_dgms)
def graphiso(dgms_list, beta, graph_isomorphism='off'):
    if graph_isomorphism == 'on':
        distinct = []
        for lis in dgms_list:
            if lis not in distinct:
                distinct.append(lis)
        print((filtration_repr(beta), len(distinct)))
        print('\n')
        # continue

@timefunction
def clf_pdvector(best_vec_result, xxx_todo_changeme, beta, Y, epd_flag = False, pvec_flag = False, vec_type = 'pi', pd_flag='False', multi_cv_flag=False, print_flag = 'off', nonlinear_flag = True, axis=1, rf_flag='y', dynamic_range_flag=True):
    # pd vector classification
        (sub_dgms, super_dgms, dgms, epd_dgms) = xxx_todo_changeme
        if pd_flag == 'True':
            import time
            if epd_flag==False:
                (stat1, stat2) = dgms_summary(dgms) # with and without multiplicity
            elif epd_flag==True:
                (stat1, stat2) = dgms_summary([add_dgms(dgms[i], epd_dgms[i]) for i in range(len(dgms))])  # with and without multiplicity

            if pvec_flag == False:
                vct_ = time.time()
                if epd_flag==False:
                    X = np.concatenate((persistence_vectors(sub_dgms, axis=axis, dynamic_range_flag=dynamic_range_flag), persistence_vectors(super_dgms, axis=1, dynamic_range_flag=dynamic_range_flag)), axis=1)
                elif epd_flag==True:
                    X = np.concatenate((persistence_vectors(sub_dgms, axis=axis, dynamic_range_flag=dynamic_range_flag),
                                        persistence_vectors(super_dgms, axis=1, dynamic_range_flag=dynamic_range_flag),
                                        persistence_vectors(epd_dgms, axis=1, dynamic_range_flag=dynamic_range_flag)),
                                       axis=1)
                vct = precision_format(time.time() - vct_, 1)
            elif pvec_flag==True: # needs to change here
                (X_sub, vct) = dgm_vec(dgms2swdgm(sub_dgms), vec_type=vec_type, axis=axis)
                (X_super, vct) = dgm_vec(dgms2swdgm(super_dgms), vec_type=vec_type, axis=axis)
                if epd_flag==False:
                    X = np.concatenate((X_sub, X_super), axis=1)
                elif epd_flag==True:
                    (X_epd, vct) = dgm_vec(dgms2swdgm(epd_dgms), vec_type=vec_type, axis=axis)
                    X = np.concatenate((X_sub, X_super, X_epd), axis=1)

            print(('Shape of X as PD vector is', (np.shape(X))))

            if rf_flag== 'y':
                rf_ = rfclf(X, Y, multi_cv_flag=multi_cv_flag);
                print(rf_)
                rf_ = ["{0:.1f}".format(100 * i) for i in rf_]
                return (rf_,(0,0))
            rf_ = rfclf(X, Y, multi_cv_flag=multi_cv_flag); # print(rf_)
            # viz_persistence_vector(dgms, Y, graph, beta, X, rf_, X_flag='Yes')
            n = len(dgms); dist_matrix = np.zeros((n, n))
            # MDS_simple(graph, beta, dist_matrix, X, Y, rf_, rs=42, annotate='no', X_flag='yes')

            t1 = time.time()
            grid_search_re = clf_search_offprint(X, Y, random_state=2, nonlinear_flag=nonlinear_flag, print_flag=print_flag)
            if grid_search_re['score'] < best_vec_result-2:
                print ('Saved one unnecessary evaluation of bad kernel ')
                return
            cv_score = evaluate_best_estimator(grid_search_re, X, Y, print_flag=print_flag)
            t2 = time.time()
            print('Finish calculating persistence diagrams\n')
            rf_ = ["{0:.1f}".format(100 * i) for i in rf_]
            cv_score = ["{0:.1f}".format(100 * i) for i in cv_score]
            return (rf_, cv_score, str(round(t2-t1)), str(grid_search_re['param']), stat1, stat2, str(vct) + ' epd_flag = ' + str(epd_flag))

def format_test():
    list = [pi, pi+1]
    list = ["{0:.3f}".format(100 * i) for i in list]
    print(list)
    print (["{0:.3f}".format(100*i) for i in list])


def ptminus(pt1, pt2):
    if str(type(pt1)) == "<class 'dionysus._dionysus.DiagramPoint'>":
        pt1 = (pt1.birth, pt1.death)
        assert type(pt1) == tuple
    if str(type(pt2)) == "<class 'dionysus._dionysus.DiagramPoint'>":
        pt2 = (pt2.birth, pt2.death)
        assert type(pt2)==tuple

    assert type(pt1) == tuple
    assert type(pt2) == tuple
    import numpy as np
    pt1_ = np.array(pt1)
    pt2_ = np.array(pt2)
    assert type(pt1_) == type(pt2_)
    return np.linalg.norm(pt1_ - pt2_)

# ptminus((1,2), (1,3))
def flippoint(pt):
    if str(type(pt)) == "<class 'dionysus._dionysus.DiagramPoint'>":
        pt = (pt.birth, pt.death)
    assert type(pt) == tuple
    assert len(pt)==2
    return (pt[1], pt[0])

def roland_kernel(dgm1, dgm2, sigma=1):
    # dgm1 = [(1,2), (2,3)]
    # dgm2 = [(1.2,2), (2.1,3)]
    # roland_kernel(dgm1, dgm2,1)
    # roland_kernel(dgm2, dgm1,1)
    # dgm1 = dgms[1]
    # dgm2 = dgms[2]
    import numpy as np
    kernel = 0
    for pt1 in dgm1:
        for pt2 in dgm2:
            kernel += np.exp(-ptminus(pt1, pt2)/8*sigma) - np.exp(-ptminus(pt1, flippoint(pt2))/8*sigma)
    return kernel/(8 * np.pi * sigma)

def gengerate_dgm(n):
    import numpy as np
    dgm = []
    for i in range(n):
        a = np.random.rand()
        b = a + np.random.rand()
        pt = (a, b)
        dgm += [pt]
    return dgm

def generate_dgms(n):
    dgms = [0] * n
    for i in range(n):
        dgms[i] = gengerate_dgm(100)
    return dgms
# dgms = generate_dgms(300)
# kernel = np.zeros((np.shape(dgms)[0], np.shape(dgms)[0]))
def get_roland_matrix_i(dgms, i, sigma=1):
    n = np.shape(dgms)[0]
    vec = np.array([0.0] * n)
    for j in range(n):
        vec[j] = roland_kernel(dgms[i], dgms[j], sigma)
    return vec

@timefunction
def get_roland_matrix_prl(dgms, sigma=1):
    from joblib import delayed, Parallel
    k = Parallel(n_jobs=-1)(delayed(get_roland_matrix_i)(dgms, i, sigma) for i in range(np.shape(dgms)[0]))
    k = np.vstack(k)
    print('Finish computing roland kernel')
    from numpy.linalg import norm
    k = k/ float(norm(k))
    return k

def get_roland_matrix(dgms):
    kernel = np.zeros((np.shape(dgms)[0], np.shape(dgms)[0]))
    n = np.shape(kernel)[0]
    for i in range(n):
        for j in range(n):
            kernel[i,j] = roland_kernel(dgms[i], dgms[j])
    return kernel

# get_roland_matrix(dgms)
# k = get_roland_matrix_prl(dgms)
# print np.all(np.linalg.eigvals(kernel) > 0)





# def normalize_kernel(K):
#     import numpy as np
#     print(np.min(K))
#     # assert np.min(K)>=0
#     # regularizer = float(0.001 * np.max(K))
#     K = K + np.eye(np.shape(K)[0]) * 1e-9
#     import numpy as np
#     assert np.shape(K)[0]==np.shape(K)[1]
#     v = np.diag(K)
#     v = np.sqrt(v) + 1e-10
#     print ('The decomposition difference is %s'%np.max((v ** 2 - np.diag(K))))
#     assert ((v**2 - np.diag(K)) < 0.001).all()
#     v = 1.0/v
#     v = np.reshape(v, ((len(v),1)))
#     assert np.shape(np.dot(v, v.T)) == np.shape(K)
#     normalized_kernel = np.multiply(K, np.dot(v, v.T))
#     print ('The symmetric difference is %s'%np.max(normalized_kernel - normalized_kernel.T))
#     # print (np.max(normalized_kernel, axis=1) == np.diag(normalized_kernel))
#     try:
#         assert (np.max(normalized_kernel, axis=1) == np.diag(normalized_kernel)).all()
#     except AssertionError:
#         print('The normalized kernel is bad')
#         print(np.sum(np.max(normalized_kernel, axis=1) == np.diag(normalized_kernel))),
#         print(np.shape(normalized_kernel))
#     # for i in range(np.shape(normalize_kernel)[0]):
#     return normalized_kernel
def normalize_kernel(K):
    assert np.shape(K)[0]==np.shape(K)[1]
    v = np.diag(K)
    v = np.sqrt(v)
    assert (v**2 - np.diag(K)).all() < 0.001
    v = 1.0/v
    v = np.reshape(v, ((len(v),1)))
    assert np.shape(np.dot(v, v.T)) == np.shape(K)
    return np.multiply(K, np.dot(v, v.T))
def gk_dist(graph):
    import numpy as np
    import graphkernels.kernels as gk
    graph = load_graph_igraph(graph)
    print((type(graph), len(graph)))
    assert list(graph.keys()) == list(range(len(graph)))
    igraph_list=[]
    # serial version
    # for i in range(len(graph)):
    #     igraph_list = igraph_list + [convert2igraph(graph[i])]

    from joblib import Parallel, delayed
    igraph_list = Parallel(n_jobs=-1)(delayed(convert2igraph)(graph[i]) for i in range(len(graph)))

    assert len(igraph_list) == len(graph)
    K_wl_2 = gk.CalculateWLKernel(igraph_list, par = 3)
    print(''); kernel_summary(K_wl_2)
    assert np.min(K_wl_2) >= 0
    K_wl_2_after = normalize_kernel(K_wl_2)
    kernel_summary(K_wl_2_after)
    return (K_wl_2, K_wl_2_after, Kernel2Dist(K_wl_2))
    # return K_wl_2_after
def MDS_simple(graph, beta, dist_matrix, X, y, rf, rs=42, annotate='no', X_flag='no'):
    # input is the distance matrix
    # ouput: draw the mds 2D embedding

    # random.seed(rs)
    np.set_printoptions(precision=4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    from sklearn import manifold

    if X_flag == 'no':
        mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
        tsne = manifold.TSNE(metric='precomputed', verbose=0, random_state=rs)
        assert np.shape(dist_matrix)[0] == np.shape(dist_matrix)[1] == len(y)
        pos = mds.fit_transform(dist_matrix)
        tsne_pos = tsne.fit_transform(dist_matrix)
    elif X_flag == 'yes':
        mds = manifold.MDS( n_jobs=-1, random_state=rs, verbose=0)
        tsne = manifold.TSNE(verbose=0, random_state=rs)
        try:
            pos = mds.fit_transform(X)
        except ValueError: # for nci1 closeness_centrality there is some error (ValueError: Array must be symmetric)
            print('There is ValueError for mds.fit')
            pos = np.zeros((np.shape(X)[0], 2))
        tsne_pos = tsne.fit_transform(X)
        pos = np.array(pos)

    for i in range(np.shape(tsne_pos)[0]):
        if tsne_pos[i,0] > -0.05:
            pass
            # break
            # print('%s,'%i),

    y_color = [color_map(i) for i in y]
    print((y_color.count('r')))
    y_color = np.array(y_color)
    assert len(y_color) == len(y)
    assert np.shape(pos)[0] == len(y); assert np.shape(tsne_pos)[0] == len(y)

    # start to plot
    fig = plt.figure()
    plt.axis('off')
    plt.title(graph + ' ' + str(beta) + '\nrf:' + str(round(100 * rf[1])) + 'BL:' + str(baseline(graph)))
    ax = fig.add_subplot(211)
    ax.scatter(np.array(pos[:, 0]), np.array(pos[:, 1]), c=y_color, s=2)
    # ax.set_ylim(-1, 1)
    # ax.set_xlim(-1, 1)
    fig.subplots_adjust(top=0.8)
    for xi, yi, pidi in zip(pos[:, 0], pos[:, 1], list(range(len(y)))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())

    ax = fig.add_subplot(212)
    ax.scatter(tsne_pos[:, 0], tsne_pos[:, 1], c=y_color, s=1.2)
    for xi, yi, pidi in zip(tsne_pos[:, 0], tsne_pos[:, 1], list(range(len(y)))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())

    direct = './Viz_algorithm/' + graph + '/vector/'
    make_direct(direct)
    filename = '/vector_' + str(beta) + '.svg'
    fig.savefig(direct + filename, format='svg')
    plt.close()
    print('Saving figure successfully')

def get_nbr_edge_vals(g, e, btwn_dict):
    assert type(e)==tuple
    assert e in list(btwn_dict.keys())
    import networkx as nx
    (u, v) = e
    nbr_edges = []
    for j in nx.neighbors(g, u):
        try:
            nbr_edges += [btwn_dict[u, j]]
        except:
            nbr_edges += [btwn_dict[j, u]]
    for j in nx.neighbors(g, v):
        try:
            nbr_edges += [btwn_dict[v, j]]
        except:
            nbr_edges += [btwn_dict[j, v]]
    return nbr_edges

def list_stat(lis):
    import numpy as np
    return {'mean': np.mean(lis), 'min': np.min(lis), 'max': np.max(lis), 'std': np.std(lis)}

def remove_none(graphs_):
    n = len(graphs_)
    result = []
    for i in range(n):
        for g in graphs_[i]:
            if g != None:
                result += [g]

def stoa():
    mlg = {'mutag': '87.4+/-1.61',  'ptc': '63.26(+/-1.48)' , 'enzyme': '61.81(+/-0.99)',  'protein_data': '76.34(+/-0.72)',  'nci1': '81.75(+/-0.24)', 'nci109': '81.31(+/-0.22)'}
    wl = {'mutag': '84.50(+/-2.16)', 'ptc': '59.97(+/-1.60)', 'enzyme': '53.75(+/-1.37)', 'protein_data': '75.49(+/-0.57)', 'nci1': '84.76(+/-0.32)', 'nci109': '85.12(+/-0.29)'}
    wl_edge = {'mutag': '82.94(+/-2.33)',  'ptc': '60.18(+/-2.19)', 'enzyme': '52.00(+/-0.72)',  'protein_data': '74.78(+/-0.59)', 'nci1': '84.65(+/-0.25)', 'nci109': '85.32(+/-0.34)'}
    fgsd = {'mutag': '92.12', 'ptc': '62.8', 'protein_data': '73.42', 'nci1': '79.8', 'nci109': '78.84', 'dd': '77.10', 'mao': '95.59',
            'reddit_binary': '86.5', 'reddit_5K': '47.76', 'reddit_12K': '47.76', 'imdb_binary': '73.62', 'imdb_multi': '52.41', 'collab': '80.02'}
    roland = {'reddit_5K': '54.5', 'reddit_12K': '44.5'}
    retgk = {'mutag': '90.31', 'ptc': '62.5', 'enzyme': '60.4', 'protein_data': '75.8', 'nci1': '84.5', 'dd': '81.6', 'collab': '81.0', 'imdb_binary': '72.3',
             'imdb_multi': '47.7', 'reddit_binary': '92.6', 'reddit_5K': '56.1', 'reddit_12K': '48.7'}
    deg_baseline={'mutag': '90.07',
                  'ptc': '61.7(50bin)/64.5(+label)/',
                  'protein_data': '71.4/72.5(50bin+cdf, 73.3 if add pair dist)/\n73.7(+label)/74.7(+label + 50bin)',
                  'nci1': '71/74.7(fine tune)',
                  'dd':'75.35/76.2(+ricci+label)/77.5(deg+dist+btwn)/\n77.5(deg+label+ricci+dist+btwn)/77.8(+ new norm)',
                  'enzyme': '35.6(+label)/38.5(+label+ricci)',
                  'reddit_binary': '90.27/91.4(+dist distribution)/\n91.6(+btwn)/92.1(cdf,100bin)',
                  'imdb_binary': '70/72.6(+edge dist and btwness)\n/74.0(+dist and btwn(300bin + 0.5ub))/ 75.4(new norm + cdf)',
                  'imdb_multi':'45(svm)/48(rf)/48.5(rf+btwn, dist feature)\n /49.0(rf+btwn, dist feature fine tunning)\n/50.0(new norm +cdf) /50.8(new norm + cc + edge feature)',
                  'reddit_5K': '53.8/54.0(+deg sum)\n /54.4(cdf + deg_sum)/54.9(log)/\n55.9(log+log 30bin)/log scale + 30 bin 55.9',
                  'reddit_12K':'43/44.0+deg sum/\n nonlinear kernel + log scale 47.8',
                  'collab': '74.7/77.0(rf)/\n77.1(+dist distribution)/\n77.6(rf + old norm +100 bin)\n 78.2(new norm + extra feature + 70 bin + cdf)'}

    return {'mlg': mlg, 'wl': wl, 'wl_edge':wl_edge, 'fgsd': fgsd, 'roland': roland, 'deg_baseline': deg_baseline, 'retgk': retgk}

def sample_data(data, Y, n_sample=200):
    import random
    n = len(Y)
    assert n > n_sample
    idx = random.sample(list(range(1, n)), n_sample)
    data = {k:data[k] for k in tuple(idx) if k in data}
    Y = Y[0:200]
    return (data, Y)
def beta_update(beta ,gradient, step=1e-1, normalization='yes'):
    print(('Step size is %s'%step))
    print(('Before update: beta is %s'%beta))
    print(('The gradient is %s'%gradient))
    beta =  (beta - step * gradient)
    if normalization == 'no':
        print(('After update: beta is %s\n' %(beta)))
        return  beta
    elif normalization == 'yes':
        print(('After update: beta is %s\n' %(beta/np.linalg.norm(beta))))
        return  beta/np.linalg.norm(beta)
    # return beta/np.sum(beta)
def print_dgm(dgm):
    for p in dgm:
        print((p), end=' ')
    print('\n')

def scatter_dgm(dgm):
    scatter = []
    for p in dgm:
        scatter += [(p.birth, p.death)]
    x = [i[0] for i in scatter]
    y = [i[1] for i in scatter]
    return scatter, x, y
def square(i):
    import dionysus as d
    import numpy as np
    f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
    m1 = d.homology_persistence(f1)
    dgms1 = d.init_diagrams(m1, f1)
    return dgms1[1]
def random_dgm(n):
    # generate ramdom dgm of cardinality n
    import numpy as np
    import dionysus as d
    lis = []
    for i in range(n):
        a = np.random.rand()
        b = np.random.rand()
        lis.append((a, a+b))
    return d.Diagram(lis)



def diag2dgm(diag):
    import dionysus as d
    if type(diag) == list:
        diag = [tuple(i) for i in diag]
    elif type(diag) == np.ndarray:
        diag = [tuple(i) for i in diag] # just help to tell diag might be an array
    dgm = d.Diagram(diag)
    return dgm

def diags2dgms(diags):
    dgms = []
    for diag in diags:
        dgms.append(diag2dgm(diag))
    return dgms

def n_distinct(dgm):
    diag = dgm2diag(dgm)
    val_list = []
    for p in diag:
        val_list += p
    return len(set(val_list))


def distinct_list(lis):
    # input: a list of lists
    # output: a list of distinct lists
    distinct = []
    for i in lis:
        # print(i)
        if (list(i) not in distinct):
            distinct.append(list(i))
    return distinct
def normalize_beta(beta):
    return beta / float(np.sum(beta))
def generate_beta(n, k=5):
    # n is the length here
    # deg, ricci, fiedler, cc, label
    import itertools
    import numpy as np
    lst = list(itertools.product(list(range(k)), repeat=n))
    # print(lst)
    weight = [np.array(tup)/float(sum(tup)) for tup in lst if (tup[0]==0 and tup[2]==0 and tup[4]==0)] # tup[4]==0 and tup[3]==0
    weight = distinct_list(weight)
    weight = [np.array(i) for i in weight]
    weight = sorted(weight, key=lambda x: x[1]) # sort by ricci
    return weight[1:] # remove [nan, nan, nan]
# weight = generate_beta(5)

def draw_landscape(graph):
    #  not real data

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    # Z = np.sin(R)
    Z = np.random.rand(40,40)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/landscape/';
    filename = 'img.png'
    plt.savefig(direct+filename)

# draw_landscape('nci1')
def beta_mesh():
    mesh={}
    X = np.arange(-1,1,0.2)
    Y = np.arange(-1, 1, 0.2)
    for i in range(len(X)):
        for j in range(len(Y)):
            mesh[(i,j)] = np.array([0, X[i], 0, Y[j], 0])
    return mesh
def dgm2diag(dgm):
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    # print(diag)
    return diag

def test_norm(dgms):
    for i in range(1, 100, 10):
        print((i), end=' ')
        print_dgm(dgms[i])
        print('\n')

@timefunction
def export_dgm(graph, dgm, i, filename = 'label', flag='ss'):
    file = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/prefiltration/'
    file = file + 'd_' + filename + '/' # change later
    make_direct(file)
    with open(file  + str(i) + '_' + flag + '.txt', 'w') as f:
        for p in dgm:
            f.write(str(p.birth) + ' ' + str(p.death) + '\n')
    f.close()


def add_row(table, pd_vector_data, name, ax, filtration_type='node'):
    if filtration_type=='empty':
        table.add_row(['']*5)
    elif filtration_type == 'node_vec':
        table.add_row([name + ' axis=' + str(ax),
                       pd_vector_data[0][0],
                       pd_vector_data[0][1] + ' (time: ' + pd_vector_data[0][2] + ')',
                       str(pd_vector_data[1][0]) + '/' + str(pd_vector_data[1][1]),
                       'vct: ' + pd_vector_data[6] +'/svm time: ' + str(pd_vector_data[2]) + ' ' + str(pd_vector_data[3]) + str(
                           pd_vector_data[4])])
    # elif filtration_type == 'node_kernel':
        # table.add_row([beta_name + ' ' + str(kernel_type) + ' bw:' + str(bandwidth), '', '',
        #                str(tda_kernel_data[0]) + '/' + str(tda_kernel_data[1]),
        #                'kpt: ' + str(t1) + ' ' + str(tda_kernel_data[2])])

    return table


def dgm2diag_(dgm):
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append((pt.birth, float('Inf')))
        else:
            diag.append((pt.birth, pt.death))
    # print(diag)
    return diag
def random_lists(n):
    import numpy as np
    lis = []
    for i in range(n):
        a = np.random.rand()
        b = np.random.rand()
        lis.append([a, a + b])
    return lis
def write_landscapedata(graph, new_data):
    assert type(new_data) == dict
    assert 'beta' in list(new_data.keys())
    assert 'others' in list(new_data.keys())

    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/landscape/';
    make_direct(direct)
    filename = 'landscape_data.txt'
    f = open(direct + filename, 'a+')
    f.write(str(new_data))
    f.write('\n')
    # f.close()

def unit_vector(n, s):
    import numpy as np
    vector = np.zeros((n))
    vector[s] = 1
    return vector
unit_vector(5,3)

# random_lists(5)
def precision_format(nbr, precision=1):
    # assert type(nbr)==float
    return  round(nbr * (10**precision))/(10**precision)


def n_runs(graph):
    if (graph == 'nci1') or (graph == 'nci109'):
        return 10
    elif (graph == 'reddit_5K') or (graph == 'reddit_12K'):
        return 10
    else:
        return 100
def progressbar():
    from progress.bar import Bar
    import time
    bar = Bar('Processing', max=10)
    for i in range(20):
        time.sleep(.1)
        next(bar)
    bar.finish()
def dgms_summary(dgms, debug='off'):
    n = len(dgms)
    total_pts = [-1]*n
    unique_total_pts = [-1] * n # no duplicates
    for i in range(len(dgms)):
        total_pts[i] = len(dgms[i])
        unique_total_pts[i] = len(set(list(dgms[i])))
    if debug == 'on':
        print('Total number of points for all dgms')
        print(dgms)
    stat_with_multiplicity = (precision_format(np.mean(total_pts), precision=1), precision_format(np.std(total_pts), precision=1), np.min(total_pts), np.max(total_pts))
    stat_without_multiplicity = (precision_format(np.mean(unique_total_pts)), precision_format(np.std(unique_total_pts)), np.min(unique_total_pts), np.max(unique_total_pts))
    print(('Dgms with multiplicity    Mean: %s, Std: %s, Min: %s, Max: %s'%(np.mean(total_pts), np.std(total_pts), np.min(total_pts), np.max(total_pts))))
    print(('Dgms without multiplicity Mean: %s, Std: %s, Min: %s, Max: %s'%(np.mean(unique_total_pts), np.std(unique_total_pts), np.min(unique_total_pts), np.max(unique_total_pts))))
    return (stat_with_multiplicity, stat_without_multiplicity)

def remove_zero_col(data, cor_flag=False, print_flag=False):
    import numpy as np
    # data = np.zeros((2,10))
    # data[1,3] = data[1,5] = data[1,7] = 1
    n_col = np.shape(data)[1]

    del_col_idx = np.where(~data.any(axis=0))[0]
    remain_col_idx = set(range(n_col)) - set(del_col_idx)
    correspondence_dict = dict(list(zip(list(range(len(remain_col_idx))), remain_col_idx)))
    inverse_correspondence_dict = dict(list(zip(remain_col_idx, list(range(len(remain_col_idx))))))

    X = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)
    if print_flag:
        print(('the shape before removing zero columns is %s' % (np.shape(data))))
        print(('the shape after removing zero columns is %s'%(np.shape(X))))
    if cor_flag == True:
        return (X, correspondence_dict, inverse_correspondence_dict)
    else:
        return X

def threshold_data(data, flag='off'):
    return 1
    # only print classify that yields reasonable good test accuracy
    if flag=='on':
        return 0
    if data == 'mutag':
        threshold = 78
    elif data == 'ptc':
        threshold = 55
    elif data == 'protein_data':
        threshold = 50
    elif data == ('nci1' or 'nci109'):
        threshold = 60
    return threshold
def onehot(n):
    import numpy as np
    y = np.zeros((2*n,2))
    for i in range(n):
        y[i][0] = -1
    for i in range(n, 2*n):
        y[i][1] = 1
    return y
def hinge_loss(y, y_hat):
    assert ((y==1) or (y==-1))
    if y == 1:
        return max(0, 1-y_hat)
    elif y== -1:
        return max(0, 1+y_hat)
def hinge_gradient(y, y_hat):
    # scalar version. The gradient of hinge loss(1-y_hat*y)+
    assert (y == 1) or (y == -1)
    if y == 1:
        if y_hat >= 1:
            return 0
        elif y_hat < 1:
            return -1
    elif y == -1:
        if y_hat >= -1:
            return 1
        elif y_hat < -1:
            return 0
# hinge_gradient(-1, -1.0003468881649533)
class parallel():
    def get_dgms_pl(self):
        from multiprocessing import Pool
        if True:
            pool = Pool(processes=2)
            list_start_vals = list(range(20))
            array_2D = pool.map(square, list_start_vals)
            pool.close()
            return array_2D
    def get_dgms_pl2(self, n_jobs=1):
        from joblib import delayed, Parallel
        return Parallel(n_jobs=n_jobs)(delayed(square)(i) for i in range(10))
class Nodeid_Unfound_Error(Exception):
    pass
class pipeline0_Error(Exception):
    pass

# for i in range(100):
#     print d.bottleneck_distance_with_edge(random_dgm(150), random_dgm(150))

##############################

class test():
    def test_kernelMachine(self):
        import numpy as np
        # X = np.array([[-1, -1], [-12, -2], [1, 1.1], [2, 1], [4, 8]])
        # y = np.array([-1, -1, 1, 1, 1])
        X = np.random.rand(100,2)
        Y = np.array([-1]*50 + [1]*50)
        n = len(Y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, list(range(n)), test_size=0.1, random_state=43)
        n_train = len(y_train)
        # kernel = pdist(X, 'euclidean')
        # kernel = squareform(kernel)
        # kernel = np.exp(np.multiply(kernel, -kernel))
        kernel = np.dot(X, X.T)
        kernel_train = kernel[np.ix_(indices_train, indices_train)]
        assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
        kernel_test = kernel[np.ix_(indices_test, indices_train)]
        np.shape(kernel_test)

        from sklearn.svm import SVC
        clf = SVC(kernel='linear', gamma=1, verbose=True)
        clf.fit(X_train, y_train)
        alpha = clf.dual_coef_
        alpha = padding_zero(alpha, n_train, clf.support_)
        # alpha = np.multiply(alpha, y_train)
        print(('Number of support vector is %s' % clf.n_support_))
        dual_regression = np.dot(kernel_test, alpha) + clf._intercept_
        primal_regression = (np.dot(X_test, clf.coef_.T) + clf.intercept_).T # clf.coef_ is only available to linear kernel
        # print('Alpha is %s' % alpha)
        # print('My Regression is %s' % regression)
        print(('Model Prediction is %s\n' % clf.predict(X_test)))

        print(('My dual Classification is %s' % np.sign(dual_regression)))
        print(('My dual primal classification is %s' % np.sign(primal_regression)))
        print(('Primal prediction is %s\n' % (np.sign(primal_regression) == clf.predict(X_test))))

        print(('My dual Prediction is %s' % regression))
        print(('My primal predication is %s'%(primal_regression)))
        print(('Decision boundary is %s'%clf.decision_function(X_test)))

        print(('Train accuracy is %s \n'%clf.score(X_train,y_train)))


        ###########
        import numpy as np
        from sklearn.svm import SVC
        from scipy.spatial.distance import pdist, squareform
        from sklearn.model_selection import train_test_split

        X = np.random.rand(100, 2)
        Y = np.array([-1] * 50 + [1] * 50)
        n = len(Y)

        clf = SVC(kernel='precomputed', verbose=True)
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, list(range(n)), test_size=0.1, random_state=43)
        n_train = len(y_train)
        kernel = pdist(X)
        kernel = squareform(kernel)
        kernel = np.exp(np.multiply(kernel, -kernel))
        kernel_train = kernel[np.ix_(indices_train, indices_train)]
        assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
        kernel_test = kernel[np.ix_(indices_test, indices_train)]

        clf.fit(kernel_train, y_train)
        alpha = clf.dual_coef_
        alpha = padding_zero(alpha, n_train, clf.support_)
        dual_regression = np.dot(kernel_test, alpha) + clf._intercept_
        # print('Alpha is %s' % alpha)
        print(('My Dual Regression is %s' % dual_regression))
        print(('Model Classification is %s' % clf.predict(kernel_test)))
        print(('Model Prediction is %s' % clf._decision_function(kernel_test).T))
        print(('Dual regression - Model prediction: %s'%(dual_regression - clf._decision_function(kernel_test).T)))
        print(('Intercept is %s'%clf.intercept_))
        ###############
        from sklearn.metrics import accuracy_score
        lbd = 1e-7
        alpha = compute_alpha_kernel(kernel_train, y_train, lbd)
        K = kernel_test
        assert np.shape(K)[1] == len(alpha)
        y_pred = np.dot(kernel_test, alpha)
        if debug == 'on':
            print(('The regression in belkin model is %s' % y_pred))
        y_reg = y_pred  # get the r_reg for gradient descent
        y_pred = np.sign(y_pred)

        y_pred_train = np.sign(np.dot(kernel_train, alpha))
        print(('Alpha is %s' % alpha))
        print(('My Regression is %s' % y_pred))
        print(('Train accuracy is %s' % accuracy_score(y_train, y_pred_train)))
        print(('Test accuracy is %s' % accuracy_score(y_test, y_pred)))
    def random_fit(self):
        def test_kernelMachine(self):
            import numpy as np
            X = np.random.rand(1000, 2)
            Y = np.array([-1] * 500 + [1] * 500)
            n = len(Y)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, list(range(n)),
                                                                                             test_size=0.01,
                                                                                             random_state=43)
            n_train = len(y_train)
            from scipy.spatial.distance import pdist, squareform
            kernel = pdist(X, 'euclidean')
            kernel = squareform(kernel)
            kernel = np.exp(np.multiply(kernel, -kernel))
            # kernel = np.dot(X, X.T)
            kernel_train = kernel[np.ix_(indices_train, indices_train)]
            assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
            kernel_test = kernel[np.ix_(indices_test, indices_train)]
            np.shape(kernel_test)

            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score
            clf = SVC(kernel='linear', gamma=1, verbose=True)
            clf.fit(X_train, y_train)
            print(('Traing accuracy is %s'%accuracy_score(clf.predict(X_train), y_train)))

    def test_wd(self):
        import dionysus as d
        import numpy as np
        f1 = d.fill_rips(np.random.random((20, 2)), 2, 1)
        m1 = d.homology_persistence(f1)
        dgms1 = d.init_diagrams(m1, f1)
        f2 = d.fill_rips(np.random.random((20, 2)), 2, 1)
        m2 = d.homology_persistence(f2)
        dgms2 = d.init_diagrams(m2, f2)
        wdist = d.wasserstein_distance(dgms1[1], dgms2[1], q=2)
        print(("2-Wasserstein distance between 1-dimensional persistence diagrams:", wdist))
        bdist = d.bottleneck_distance(dgms1[1], dgms2[1])
        print(("Bottleneck distance between 1-dimensional persistence diagrams:", bdist))

    def test_wd_symmetric(self):
        import dionysus as d
        dgm1 = d.Diagram([(5.5, 4.68182), (8.04545, 4.59091), (2,2.0001)])
        dgm2 = d.Diagram([(5.96, 3.48), (5.24, 4.04), (5.24, 4.04), (4.44, 3.48)])
        print((d.bottleneck_distance(dgm1, dgm2)))
        print((d.bottleneck_distance(dgm2, dgm1)))
        print((d.bottleneck_distance()))

    def test_bd_with_edge(self):
        import dionysus as d
        # dgm1 = d.Diagram([(5.5, 4.68182), (8.04545, 4.59091)])
        # dgm2 = d.Diagram([(5.96, 3.48), (5.24, 4.04), (5.24, 4.04), (4.44, 3.48)])
        # dgm1 = d.Diagram([(4.68182, 5.5), (4.59091, 8.04545)])
        # dgm2 = d.Diagram([(3.48, 5.96), (4.04, 5.24), (4.04, 5.24), (3.48, 4.44)])

        dgm1 = d.Diagram([(1.1, 0)])
        dgm2 = d.Diagram([(0.1, 0)])
        print((d.bottleneck_distance_with_edge(dgm1, dgm2)))


# new_label('protein_data')
def check_edge_filtration(allowed):
    if ('jaccard' in allowed) or ('edge_p' in allowed) or ('jaccard_int' in allowed) or ('ricci_edge' in allowed):
        # assert len(allowed)==1
        return True
    else:
        return False
def change_label_(y, new_label):
    y_old = [1] * len(y)
    for idx in new_label:
        y_old[idx] = -1

    return np.array(y_old).reshape(len(y),)

def change_label(graph, Y, change_flag='no', false_label_percent=0):
    Y = Y.copy()
    label = list(set(Y))
    import numpy as np
    np.random.seed(42)
    idx = np.random.choice(list(range(len(Y))), size=int(len(Y)*false_label_percent), replace=False)
    assert len(idx) == len(set(idx))
    if len(idx)==0:
        return Y
    print(idx)
    for i in idx:
        label_tmp = [x for x in label if x != Y[i]]
        Y[i] = np.random.choice(label_tmp)
    if change_flag == 'no':
        return Y
    elif change_flag == 'yes':
        if (graph == 'protein_data') or (graph == 'mutag'):
            Y = change_label_(Y, new_label(graph))
            return Y


def check_kernel():

    pass
def graphsimport():
    np.set_printoptions(precision=4)
    import matplotlib
    matplotlib.use('Agg')
    # matplotlib.use('Agg')


# SVM realted
def compute_alpha_kernel(kernel, Y, lbd):
    import numpy as np
    n = np.shape(Y)[0]
    inv_K = np.linalg.inv(kernel + lbd * np.eye(n))
    alpha = np.dot(inv_K, Y)
    assert len(alpha) == n
    return alpha
class kernelMachine():
    def __init__(self, kernel, c, Y, lbd = 1e-7, loss='hinge', debug='off'):
        self.kernel = kernel / np.linalg.norm(kernel, 2)
        self.c = c
        self.n = np.shape(Y)[0]
        self.Y = Y.reshape(self.n,1)
        self.lbd = lbd
        self.loss = loss
        self.debug= debug
        self.test_accuracies = []
        self.train_accuracies = []

    def compute_alpha_kernel(self):
        import numpy as np
        n = np.shape(self.Y)[0]
        inv_K = np.linalg.inv(self.kernel + self.lbd * np.eye(n))
        alpha = np.dot(inv_K, self.Y)
        assert len(alpha) == n
        return alpha

    def get_alpha(self, clf, n):
        # get the dual coefficient
        # X = np.array([[-1, -1], [-12, -2], [1, 1], [2, 1], [4, 8]])
        #         y = np.array([0, 0, 1, 1, 1])
        #         from sklearn.svm import SVC, SVR
        #         clf = SVC(gamma=1)
        #         clf.fit(X, y)
        #
        alpha = np.zeros(n)
        sprt_idx = clf.support_
        dual_coef = clf.dual_coef_
        k = 0
        for i in sprt_idx:
            alpha[i]= dual_coef[0][k] # need to change if want to handle multiple classes
            k = k + 1
        if self.debug=='on':
            print(dual_coef)
            print((clf.n_support_))
            print((k, np.sum(clf.n_support_)))
        # assert k ==np.sum(clf.n_support_)
        assert  k == len(clf.support_)
        return alpha

    def single_test(self):
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        loss = self.loss
        kernel = self.kernel
        n = self.n
        Y = self.Y
        c = self.c

        print(('.'), end=' ')
        y_train, y_test, indices_train, indices_test = train_test_split(Y, list(range(n)), test_size=0.1, random_state=i)
        n_train = len(y_train)
        # kernel = np.exp(np.multiply(dist_matrix, -dist_matrix) / sigma)
        kernel_train = kernel[np.ix_(indices_train, indices_train)]
        assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
        kernel_test = kernel[np.ix_(indices_test, indices_train)]
        np.shape(kernel_test)

        if loss == 'hinge':
            clf = svm.SVC(kernel='precomputed', C=c)
            clf.fit(kernel_train, y_train)
            y_pred = clf.predict(kernel_test)
            y_reg = y_pred # need to change later
            alpha = self.get_alpha(clf, n_train)

            y_pred_train = clf.predict(kernel_train)
            # print('Train accuracy is %s'%accuracy_score(y_train, y_pred_train))

        elif loss == 'square':
            lbd = self.lbd
            alpha = self.compute_alpha_kernel(kernel_train, y_train, lbd)
            K = kernel_test
            assert np.shape(K)[1] == len(alpha)
            y_pred = np.dot(kernel_test, alpha)
            if self.debug=='on':
                print(('The regression in belkin model is %s'%y_pred))
            y_reg = y_pred # get the r_reg for gradient descent
            y_pred = np.sign(y_pred)
            # y_pred = ((y_pred + 1) / 2).astype(int)

            y_pred_train = np.sign(np.dot(kernel_train, alpha))
            # print('Train accuracy is %s' % accuracy_score(y_train, y_pred_train))

        elif loss == 'svr':
            clf = svm.SVR(kernel='precomputed', C=c)
            clf.fit(kernel_train, y_train)
            y_pred = clf.predict(kernel_test)
            y_reg = y_pred
            y_pred = np.sign(y_pred)
            alpha = self.get_alpha(clf, n_train)

            y_pred_train = np.sign(clf.predict(kernel_train))
            # print('Train accuracy is %s' % accuracy_score(y_train, y_pred_train))

        self.test_accuracies.append(accuracy_score(y_test, y_pred))
        self.train_accuracies.append(accuracy_score(y_train, y_pred_train))
        assert 0 not in self.test_accuracies
        assert 0 not in self.train_accuracies

    def multi_tests(self):
        for i in range(10):
            self.single_test()
        print(('Loss: %s, Training accuracy is %s' % (self.loss, np.mean(self.train_accuracies))))
        if (self.loss == 'hinge') or (self.loss == 'svr'):
            print(('c is %s, mean accuracy is %s, std is %s \n' % (self.c, np.mean(self.test_accuracies), np.std(self.test_accuracies))))
        elif self.loss == 'square':
            print(('lambda is %s, mean accuracy is %s, std is %s \n' % (
            self.lbd, np.mean(self.test_accuracies), np.std(self.test_accuracies))))

        # return {'reg': self.y_reg, 'test_idx': self.indices_test, 'train_idx': indices_train, 'coef': alpha}
# km = kernelMachine(kernel=kernel, c=1, Y=Y)

# import dionysus as d
# import numpy as np
# diag1 = np.array([[1,2],[2,3],[6,8]])
# diag2 = np.array([[1.1,2.8],[2.2,3.9],[6.1,20]])
# diag1 = [[1,2],[2,3],[6,8]]
# diag2 = [[1.1,2.8],[2.2,3.9],[6.1,20]]
#
# diag1 = random_lists(50)
# diag2 = random_lists(20)
# dgm1 = d.Diagram([(p[0], p[1]) for p in diag1]) # convert a list of lists to a list of tuples
# dgm2 = d.Diagram([(p[0], p[1]) for p in diag2])
# bd = d.bottleneck_distance(dgm1, dgm2, delta=0.001)
# print d.bottleneck_distance_with_edge(dgm1, dgm2, delta=0.001)
# d_bottleneck_distance_with_edge_base(diag1, diag2, bd=bd)
#
#
def padding_zero(alpha, n, support):
    # take dual coefficeitn and pad zero
    # input: np.array; expect clf.support_ that gives the location of nonzero coefficient
    assert np.shape(alpha)[1] == len(support)
    lis = [0] * n
    for i in range(len(support)):
        idx = support[i]
        # print('right is %s'% alpha[0][i])
        lis[idx] = alpha[0][i]
    lis = np.array(lis)
    assert len(lis) == n
    return lis

##############################

####################################################
# Homology computation
def find_cycle(g, node, debug='on'):
    from networkx import cycle_basis
    basis_list = cycle_basis(g)
    print(('Number of basis is %s' % len(basis_list)))
    homology_list = []
    # print('New node is %s'%node[0])
    # print('Basis list is %s'%basis_list)
    for lis in basis_list:
        if node[0] in lis:
            birth = birth_comp(lis, g, sub=True)
            death = birth_comp(lis, g, sub=False)
            # print(lis),
            # print(birth),
            # print(death)
            homology_list.append({'birth': birth, 'death': death, 'potential_good_basis': lis})
        else:
            homology_list.append({'birth': None, 'death': None, 'bad_basis': lis})
            if debug == 'on':
                print(('The old cycle is %s' % lis))

    if homology_list != []:
        return homology_list
def node_filter(g, threshold, sub=True):
    # output node values below/above certain threshold
    nodesless = []
    if sub == True:
        for (n, d) in g.nodes(data=True):
            if d['fv_random'] <= threshold:
                assert g.node[n]['fv_random'] <= threshold
                nodesless.append(n)
    elif sub == False:
        for (n, d) in g.nodes(data=True):
            if d['fv_random'] >= threshold:
                assert g.node[n]['fv_random'] >= threshold
                nodesless.append(n)

    return nodesless
def cycle_plus(lis1, lis2):
    # needs refinment
    # make sure the common insersection in lis1 is increasing
    import numpy as np
    assert type(lis1) == type(lis2) == list
    assert len(lis1) == len(np.unique(lis1))
    assert len(lis2) == len(np.unique(lis2))

    common = list(set(lis1) & set(lis2))
    idx1 = [lis1.index(i) for i in common]
    idx2 = [lis2.index(i) for i in common]
    idx1_diff = [t - s for s, t in zip(idx1, idx1[1:])]
    if -1 in idx1_diff:
        lis1.reverse()
        idx1 = [lis1.index(i) for i in common]
        idx1_diff = [t - s for s, t in zip(idx1, idx1[1:])]
    assert -1 not in idx1_diff
    idx2_diff = [t - s for s, t in zip(idx2, idx2[1:])]
    assert len(idx1_diff) == len(idx2_diff)
    new_cycle = []
    if idx1_diff == idx2_diff == [1] * len(idx1_diff):
        i = idx1[-1]
        while i % len(lis1) != idx1[0]:
            new_cycle.append(lis1[i % len(lis1)])
            i = i + 1

        i = idx2[0]
        while i % len(lis2) != idx2[-1]:
            new_cycle.append(lis2[i % len(lis2)])
            i = i - 1

    elif idx1_diff == [-i for i in idx2_diff] == [1] * len(idx1_diff):
        i = idx1[-1]
        while i % len(lis1) != idx1[0]:
            new_cycle.append(lis1[i % len(lis1)])
            i = i + 1

        i = idx2[0]
        while i % len(lis2) != idx2[-1]:
            new_cycle.append(lis2[i % len(lis2)])
            i = i + 1
    else:
        raise AssertionError
    return new_cycle
# lis1 = [19, 20, 21, 22, 3, 4]
# lis1.reverse()
# lis2 = [1, 2, 3, 4, 5, 6, 15, 14, 13, 12, 11, 10, 9, 0]
# cycle_plus(lis1, lis2)
def merge_two_dicts(x, y):
    assert type(x) == type(y) == dict
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z
def get_pers_data(g, fv_list, order='increasing'):
    pers_data = list()
    if order == 'increasing':
        rangelist = list(range(len(fv_list)))
    elif order == 'decreasing':
        rangelist = list(range(len(fv_list), -1, -1))

    for i in rangelist:
        import networkx as nx
        threshold = fv_list[i]
        if i == 0:
            diff = set(node_filter(g, threshold, False))
            newnode = list(diff)
        elif i > 0:
            old_threshold = fv_list[i - 1]
            diff = set(node_filter(g, threshold, False)) - set(node_filter(g, old_threshold, False))
            newnode = list(diff)
            print(('Threshold is %s, New nodes are %s' % (threshold, newnode)))

        g_sub = g.subgraph(node_filter(g, threshold, False))
        component = [list(i) for i in list(sorted(nx.connected_components(g_sub)))]
        print(('Number of component is %s, The component lists are: %s' % (len(component), component)), end=' ')
        print('\n')
        homology1_list = find_cycle(g_sub, newnode)
        homology0_list = [{'Threshold': threshold, 'Node': list(diff), 'N_Comp': len(component), 'Comp': component}]
        pers_data = pers_data + [homology0_list]
        # pers_data = pers_data + [homology0_list, homology1_list]
        print((g_sub.nodes()))
        print((g_sub.edges()))

    return pers_data
def diff(first, second):
    # first: set of sets
    first = set(frozenset(i) for i in first)
    second = set(frozenset(i) for i in second)
    second = set(second)
    return {'d2': [list(item) for item in second if item not in first],
            'd1': [list(item) for item in first if item not in second]}
def birth_comp(lis, g, sub=True):
    if (sub == True) or (sub == 'sub') or (sub == 'min'):
        birth = 1e5
        for i in lis:
            if birth > g.node[i]['fv']:
                birth = g.node[i]['fv']
        return birth
    elif (sub == False) or (sub == 'super') or (sub == 'max'):
        birth = -1e5
        for i in lis:
            if birth < g.node[i]['fv']:
                birth = g.node[i]['fv']
        return birth
def zero_homology(pers_data):
    for i in range(len(pers_data) - 1):
        print('\n')
        difference = diff(pers_data[i][0]['Comp'], pers_data[i + 1][0]['Comp'])
        print((difference), end=' ')
        print((pers_data[i + 1][0]['Node']))
        newnode = pers_data[i + 1][0]['Node'][0]
        b_list = difference['d1']
        a_list = difference['d2']

        if len(b_list) == 0:
            assert len(a_list) == 1
            assert len(a_list[0]) == 1
            # assert b_list[0] == pers_data[i+1]['Node']
            print(('New node is a new component: Node %s, value %s ' % (
            pers_data[i][0]['Node'], pers_data[i][0]['Threshold'])))
            # print('The birth time of new node %s is %s'%(newnode ,g.node[newnode]['fv_random']))
        elif len(b_list) == 1:
            if a_list[0].sort() == list((b_list[0] + pers_data[i + 1][0]['Node'])).sort():
                print('New node merged in the existing component')

        elif len(b_list) == 2:
            if a_list[0].sort() == list((b_list[0] + b_list[1] + pers_data[i + 1][0]['Node'])).sort():
                print(('New node connect two existing components %s and %s' % (b_list[0], b_list[1])))
                if birth_comp(b_list[0], g, False) < birth_comp(b_list[1], g, False):
                    print(('The birth time of %s is %s' % (b_list[0], birth_comp(b_list[0], g, 'max'))))
                    print(('The death time of %s is %s' % (b_list[0], g.node[newnode]['fv_random'])))

                elif birth_comp(b_list[0], g, False) >= birth_comp(b_list[1], g, False):
                    print(('The birth time of %s is %s' % (b_list[1], birth_comp(b_list[1], g, 'max'))))
                    print(('The death time of %s is %s' % (b_list[1], g.node[newnode]['fv_random'])))

                else:
                    print('Unconsidered cases')
                    raise Exception

        elif len(b_list) == 3:
            print('One node kills more than one component')
            raise AssertionError
        else:
            print('Unconsiderd cases')
def get_matrix(dgms, dtype='bottleneck'):
    # get dist matrix from dgms(a list of pds)
    # serial computing
    import numpy as np
    import dionysus as d
    assert type(dgms) == list
    n = len(dgms)
    dist_matrix = np.zeros((n, n))+ 1.111
    idx_dict = {}
    if dtype == 'bottleneck':
        for i in range(n):
            for j in range(n):
                dist = d.bottleneck_distance_with_edge(dgms[i], dgms[j])
                print(dist)
                dist_matrix[i, j] = dist[0]
                (idx1, idx2) = dist[1]
                if ((str(idx1)=='-1') and (str(idx2)=='-1')):
                    idx_dict[(i, j)] = 'same'
                elif ((str(idx1)!='-1') and (str(idx2)!='-1')):
                    idx_dict[(i, j)] = ((dgms[i][idx1].birth, dgms[i][idx1].death), (dgms[j][idx2].birth, dgms[j][idx2].death))
                elif ((str(idx1)!='-1') and (str(idx2)=='-1')) :
                    idx_dict[(i, j)] = ((dgms[i][idx1].birth, dgms[i][idx1].death), None)
                elif ((str(idx1)=='-1') and (str(idx2)!='-1')):
                    idx_dict[(i, j)] = (None, (dgms[j][idx2].birth, dgms[j][idx2].death))
                else:
                    raise AssertionError
                    # assert np.amax(abs((dist_matrix - dist_matrix.T))) < 0.02 # db is not symmetric
        return ((dist_matrix + dist_matrix.T) / 2.0, idx_dict)
    elif dtype == 'wasserstein':
        for i in range(n):
            for j in range(n):
                import dionysus as d
                dist = d.wasserstein_distance(dgms[i], dgms[j], q=2)
                dist_matrix[i, j] = dist
        # assert np.amax(abs((dist_matrix - dist_matrix.T))) < 0.02 # db is not symmetric
        return (dist_matrix + dist_matrix.T) / 2.0

def test_dionysus_modification():
    simplices = [([2], 4), ([1,2], 5), ([0,2], 6),
                  ([0], 1),   ([1], 2), ([0,1], 3)]
    f = d.Filtration()
    for vertices, time in simplices:
         f.append(d.Simplex(vertices, time))

    def compare(s1, s2, sub_flag=True):
        if sub_flag==True:
            if s1.dimension()>s2.dimension():
                return 1
            elif s1.dimension()<s2.dimension():
                return -1
            else:
                return cmp(s1.data, s2.data)
        elif sub_flag==False:
            return -compare(s1, s2, sub_flag=True)

    f.sort(cmp=compare)
    for s in f:
        print(s)

