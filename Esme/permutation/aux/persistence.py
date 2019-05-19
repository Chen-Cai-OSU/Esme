import random;

random.seed(42)
import argparse
# sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode')
# sys.path.append('/home/cai.507/Documents/Utilities/dionysus/build/bindings/python')
import dionysus as d
import numpy as np
from Esme.helper.time import timefunction
from Esme.permutation.aux.cycle_tools import load_data
# from helper import *
# from localtest import *
from Esme.permutation.aux.test0 import high_order
from Esme.permutation.aux.testground import searchclf, evaluate_clf
import networkx as nx
from joblib import Parallel, delayed


def coarse_pipeline0(i, allowed, version=1, print_flag='False', norm_flag='no', feature_addition_flag=False):
    # i = 1; allowed = ['deg']; version=1; print_flag='False'; norm_flag='no'; feature_addition_flag=False
    result = []
    for j in range(i, i + 50):
        if j >= n:
            continue
        tmp = pipeline0(j, allowed, version, print_flag=print_flag, norm_flag=norm_flag,
                        feature_addition_flag=feature_addition_flag)
        result = result + [tmp]
    return result


@timefunction
def quick_compute_graphs_(allowed, norm_flag='yes', feature_addition_flag=False):
    from joblib import Parallel, delayed
    result = Parallel(n_jobs=-1)(
        delayed(coarse_pipeline0)(i, allowed, norm_flag=norm_flag, feature_addition_flag=feature_addition_flag) for i in
        range(0, n, 50))
    result = [gs for res in result for gs in res]
    assert len(result) == n
    return result


def pipeline0(i, allowed, version=1, print_flag='False', norm_flag='no', feature_addition_flag=False):
    # basically two steps. 1) convert data dict to netowrkx graph 2) calculate function on networkx graphs
    # bar.next()
    print(('*'), end=' ')
    # version1: deal with chemical graphs
    # version2: deal with all non-isomorphic graphs
    # prepare data. Only execute once.
    if version == 1:
        assert 'data' in globals()
        if not feature_addition_flag:
            gi = convert2nx(data[i], i)
    elif version == 2:
        gi = nx_graphs_[i][0]

    if not feature_addition_flag:
        subgraphs = get_subgraphs(gi)
    elif feature_addition_flag:
        assert 'graphs_' in list(globals().keys())
        # global graphs_
        subgraphs = [g.copy() for g in graphs_[i]]

    gi_s = [function_basis(gi, allowed, norm_flag=norm_flag) for gi in subgraphs]
    gi_s = [g for g in gi_s if g != None]
    if print_flag == 'True':
        pass
        # print('graph %s, n_nodes: %s, n_edges: %s',%(i,len(gi_s[0]),len(gi_s[0].edge)))
    # print('OS: %s, Graph %s: Pipeline1 Finishes'%(os.getpid(), i))
    return gi_s


def pipeline1(i, beta=np.array([0, 0, 0, 0, 1]), hop_flag='n', basep=0, debug='off', rs=100,
              edge_fil='off'):  # beta= [deg, ricci, fiedler, cc]
    # data: mutag dict
    # calculate persistence diagram of graph(may disconneced)
    import dionysus as d
    if (i % 50 == 0):
        print(('.'), end=' ')
    if debug == 'on':
        print(('Processing %s' % i))
    assert 'data' in globals()
    dgm_ = d.Diagram()
    subgraphs = [];
    dgm_ = d.Diagram([(0, 0)]);
    dgm_sub = d.Diagram([(0, 0)]);
    dgm_super = d.Diagram([(0, 0)]);
    epd_dgm = d.Diagram([(0, 0)])
    for k in range(len(graphs_[i])):
        if debug == 'on':
            print(('Processing subgraph %s' % k))

        g = graphs_[i][k]
        assert str(
            type(g)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        g = fv(g, beta, hop_flag=hop_flag, basep=basep, rs=rs, edge_fil=edge_fil)  # belong to pipe1
        (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='max')  # belong to pipe1
        dgm_sub = get_diagram(g, key='fv', subflag='True')

        (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='min')  # belong to pipe1
        dgm_super = get_diagram(g, key='fv', subflag='False')
        dgm_super = flip_dgm(dgm_super)
        epd_dgm = get_diagram(g, key='fv', one_homology_flag=True)

        dgm = add_dgms(dgm_sub, dgm_super)
        if debug == 'on':
            print(('Individual dgm:'), end=' ')
            print_dgm(dgm)
        dgm_ = add_dgms(dgm_, dgm)
        subgraphs.append(g)

    if debug == 'on':
        print(('Final dgm:'), end=' ')
        print_dgm(dgm_)
    if i % 100 == 0:
        print_dgm(dgm)
    return (subgraphs, dgm_, dgm_sub, dgm_super, epd_dgm)


def handle_i_component(i, rs, component=0, debug_flag='off'):
    g = graphs_[i][component];
    n_base = len(g)
    # assert hop_flag == 'y'
    subgraphs_b = [];
    dgm_b = d.Diagram()
    dgm_sub_b = d.Diagram()
    dgm_super_b = d.Diagram()
    for basep in g.nodes():
        # basep = 1
        (subgraphs, dgm_, dgm_sub, dgm_super, epd_dgm) = pipeline1(i, beta=beta, hop_flag='y', basep=basep, rs=rs)
        dgm_b = add_dgms(dgm_b, dgm_)
        dgm_sub_b = add_dgms(dgm_sub_b, dgm_sub)
        dgm_super_b = add_dgms(dgm_super_b, dgm_super)
        if debug_flag == 'on':
            print(('new dgms has', len(dgm_)), end=' ')
            print(('total dgm_b has ', len(dgm_b)))

    return (None, dgm_b, dgm_sub_b, dgm_super_b)


def handle_i(i, rs, debug_flag='off'):
    n = len(graphs_[i])
    dgm_ = d.Diagram()
    dgm_sub = d.Diagram()
    dgm_super = d.Diagram()
    for k in range(n):
        (_, dgm_b, dgm_sub_b, dgm_super_b) = handle_i_component(i, rs, component=k, debug_flag=debug_flag)
        dgm_ = add_dgms(dgm_b, dgm_)
        dgm_sub_ = add_dgms(dgm_sub_b, dgm_sub)
        dgm_super_ = add_dgms(dgm_super_b, dgm_super)
    return (None, dgm_, dgm_sub_, dgm_super_)


# @timefunction
def get_dgms(beta=np.array([0, 0, 0, 0, 1]), parallel='on', n_jobs=-1, hop_flag='n', basep=0, rs=100, edge_fil='off'):
    # run pipeline1 in parallel
    import time
    start = time.time()
    assert 'graphs_' in list(globals().keys())
    n = len(graphs_)
    # data is dict
    if parallel == 'off':
        dgms = [0] * n
        graphs = [0] * n
        sub_dgms = [0] * n
        super_dgms = [0] * n
        epd_dgms = [0] * n
        for i in range(n):
            (graphs[i], dgms[i], sub_dgms[i], super_dgms[i], epd_dgms[i]) = pipeline1(i, beta=beta, hop_flag=hop_flag,
                                                                                      basep=basep, rs=rs,
                                                                                      edge_fil=edge_fil)
        assert 0 not in dgms
        return [(graphs[i], dgms[i], sub_dgms[i], super_dgms[i], epd_dgms[i]) for i in range(n)]  # tuple of length 2

    elif parallel == 'on':
        if basep == 'a':
            assert hop_flag == 'y'
            X = Parallel(n_jobs=-1)(delayed(handle_i)(i, rs) for i in range(n))  # a list of tuples
            return X
            # beta = np.array([1,0,0,0,0]); hop_flag = 'y'; rs=100
            # handle_i_component(10)
            # handle_i_component(11)
            # handle_i(100, debug_flag='off')

        return Parallel(n_jobs=n_jobs)(
            delayed(pipeline1)(i, beta=beta, hop_flag=hop_flag, basep=basep, rs=rs, edge_fil=edge_fil) for i in
            range(n))


def dgms_data(graph, beta, n_jobs, debug_flag, norm_flag='no', hop_flag='n', basep=-1, rs=100, edge_fil='off'):
    # for beta, get the corresponding dgm(sub/super dgms)
    # save it for future use if necessary
    if basep > 0:
        print(('Using base point %s' % basep))
    global graphs_
    # assert type(beta) is np.ndarray # needs to relax to accompodate 'hop_c'
    (graphs, flag1) = load_data(graph, 'graphs', beta, no_load='yes')
    (dgms_, flag2) = load_data(graph, 'dgms_', beta, no_load='yes')
    if flag2 == 'success':
        dgms = Parallel(n_jobs=-1)(delayed(diag2dgm)(dgms_[i]) for i in range(len(dgms_)))

    if (flag1 != 'success') or (flag2 != 'success'):
        print('Computing graphs and dgms...')
        databundle = get_dgms(beta=beta, hop_flag=hop_flag, basep=basep, rs=rs,
                              edge_fil=edge_fil)  # haven't handle parallel case
        (graphs, dgms, sub_dgms, super_dgms, epd_dgms) = unzip_databundle(databundle)
        # dump data
        dump_data(graph, graphs, 'graphs', beta, skip='yes')
        for tmp in ['dgms', 'sub_dgms', 'super_dgms']:
            save_data = eval(tmp)
            data_ = Parallel(n_jobs=-1)(delayed(dgm2diag)(save_data[i]) for i in range(len(save_data)))
            dump_data(graph, data_, tmp, beta)

    return (graphs, dgms, sub_dgms, super_dgms, epd_dgms)


def getbaselineX(i):
    from .testground import attributes
    assert 'graphs_' in globals()
    g = graphs_[i][0]
    return attributes(g)


def function_basis_test():
    # assert efficient recomputation
    # function_basis don't recompute the same function every time
    def getG():
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([1, 3, 5, 7, 8, 10])
        G.add_edges_from([(1, 3), (1, 5), (1, 7), (5, 7), (8, 3), (8, 5), (10, 5), (10, 7)])
        return G

    g = getG()
    g_cc = function_basis(g, ['cc'], norm_flag='no');
    print((g_cc.node[1]))
    g_cc_deg = function_basis(g_cc, ['fiedler']);
    print((g_cc_deg.node[1]))
    g_ = function_basis(g, ['cc', 'fiedler']);
    print((g_.node[1]))
    for v in g_.nodes():
        assert g_.node[v] == g_cc_deg.node[v]


def f_transform(x, param={}, type_='poly'):
    # transform filtration function x -> f(x) where f can be polynomial
    if type == 'log':
        import numpy as np
        return np.log(x + 1)
    if type_ == 'poly':
        assert type(param) == dict
        return param['a'] * x ** 2 + param['b'] * x + param['c']
    if type_ == 'identity':
        return x


# @timefunction


# @profile
def function_basis(g, allowed, norm_flag='no', recomputation_flag=False, transformation_flag=True):
    # function_basis.counter +=1
    # print function_basis.counter,

    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    # allowed = ['ricci', 'deg', 'hop', 'cc', 'fiedler']

    # to save recomputation. Look at the existing feature at first and then simply compute the new one.
    if len(g) < 3:
        return
    existing_features = [list(g.node[list(g.nodes())[0]].keys())]
    if not recomputation_flag:
        allowed = [feature for feature in allowed if feature not in existing_features]
    elif recomputation_flag:
        allowed = allowed
    # print('Recompute only those features', allowed)
    import networkx as nx
    import numpy as np
    assert nx.is_connected(g)
    from Esme.graph.OllivierRicci import ricciCurvature
    def norm(g, key, flag=norm_flag):
        if flag == 'no':
            return 1
        elif flag == 'yes':
            return np.max(np.abs(list(nx.get_node_attributes(g, key).values()))) + 1e-6
            # return 0.01
            # get the max of g.node[i][key], has some problem actually
            for v, data in sorted(g.node(data=True), key=lambda x: abs(x[1][key]), reverse=True):
                norm = np.float(data[key]) + 1e-6
                return norm

    # ricci
    g_ricci = g
    if 'ricciCurvature' in allowed:
        try:
            g_ricci = ricciCurvature(g, alpha=0.5, weight='weight')
            assert list(g_ricci.node.keys()) == list(g.nodes())
            ricci_norm = norm(g, 'ricciCurvature', norm_flag)
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] /= ricci_norm
        except:
            print('RicciCurvature Error for graph, set 0 for all nodes')
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] = 0
            # don't know what I wrote the follwing code beore
            # ricci_norm = norm(g, 'ricciCurvature', norm_flag)
            # for n in g_ricci.nodes():
            #     g_ricci.node[n]['ricciCurvature'] /= ricci_norm

    # degree
    if 'deg' in allowed:
        deg_dict = dict(nx.degree(g_ricci))
        for n in g_ricci.nodes():
            g_ricci.node[n]['deg'] = deg_dict[n]
            # g_ricci.node[n]['deg'] = np.log(deg_dict[n]+1)

        # deg_norm = np.float(max(deg_dict.values()))
        deg_norm = norm(g_ricci, 'deg', norm_flag)
        for n in g_ricci.nodes():
            g_ricci.node[n]['deg'] /= np.float(deg_norm)

        # return g_ricci

        # hop

    # var
    if 'var' in allowed:
        import scipy
        distance = nx.floyd_warshall_numpy(g);  # return a matrix
        distance = np.array(distance)
        distance = distance.astype(int)
        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            assert n_idx <= len(g_ricci)
            g_ricci.node[n]['var1'] = np.mean(distance[n_idx])
            g_ricci.node[n]['var'] = scipy.stats.moment(distance[n_idx], 2)
            g_ricci.node[n]['var4'] = scipy.stats.moment(distance[n_idx], 4)
        norm_var1 = norm(g_ricci, 'var1', norm_flag)
        norm_var = norm(g_ricci, 'var', norm_flag)
        norm_var4 = norm(g_ricci, 'var4', norm_flag)

        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['var1'] /= np.float(norm_var1)
            g_ricci.node[n]['var'] /= np.float(norm_var)
            g_ricci.node[n]['var4'] /= np.float(norm_var4)

    if 'hop' in allowed:
        distance = nx.floyd_warshall_numpy(g)  # return a matrix
        distance = np.array(distance)
        distance = distance.astype(int)
        if norm_flag == 'no':
            hop_norm = 1
        elif norm_flag == 'yes':
            hop_norm = np.max(distance)
        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            assert n_idx <= len(g_ricci)
            # print(n, n_idx)
            g_ricci.node[n]['hop'] = distance[n_idx][:] / float(hop_norm)

    if 'hop_' in allowed:
        distance = nx.floyd_warshall_numpy(g)  # return a matrix
        distance = np.array(distance)
        distance = distance.astype(int)
        if norm_flag == 'no':
            hop_norm = 1
        elif norm_flag == 'yes':
            hop_norm = np.max(distance)
        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            assert n_idx <= len(g_ricci)
            for n_ in g_ricci.nodes():
                n__idx = list(g_ricci.nodes).index(n_)
                g_ricci.node[n]['hop_' + str(n_)] = distance[n__idx][n_idx] / float(hop_norm)

    # closeness_centrality
    if 'cc' in allowed:
        closeness_centrality = nx.closeness_centrality(g)  # dict
        closeness_centrality = {k: v / min(closeness_centrality.values()) for k, v in
                                closeness_centrality.items()}  # no normalization for debug use
        closeness_centrality = {k: 1.0 / v for k, v in closeness_centrality.items()}
        for n in g_ricci.nodes():
            g_ricci.node[n]['cc'] = closeness_centrality[n]

    # fiedler
    if 'fiedler' in allowed:
        from networkx.linalg.algebraicconnectivity import fiedler_vector
        fiedler = fiedler_vector(g, normalized=False)  # np.ndarray
        assert max(fiedler) > 0
        fiedler = fiedler / max(np.abs(fiedler))
        assert max(np.abs(fiedler)) == 1
        for n in g_ricci.nodes():
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['fiedler'] = fiedler[n_idx]

    if 'e1' in allowed:
        import scipy
        eigvs = scipy.linalg.eig(nx.laplacian_matrix(g).todense())[1]
        if len(eigvs) <= 6:
            n = len(eigvs)
            first3 = np.array([[1] * n] * 3)
            last3 = np.array([[1] * n] * 3)
        else:
            first3 = eigvs[0:3]
            last3 = eigvs[-3:]
        for n in g_ricci.nodes():
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['e1'] = first3[0][n_idx] / max(abs(first3[0]))
            g_ricci.node[n]['e2'] = first3[1][n_idx] / max(abs(first3[1]))
            g_ricci.node[n]['e3'] = first3[2][n_idx] / max(abs(first3[2]))
            g_ricci.node[n]['e-1'] = last3[0][n_idx] / max(abs(last3[0]))
            g_ricci.node[n]['e-2'] = last3[1][n_idx] / max(abs(last3[1]))
            g_ricci.node[n]['e-3'] = last3[2][n_idx] / max(abs(last3[2]))

    any_node = list(g_ricci.node)[0]
    if 'label' not in list(g_ricci.node[any_node].keys()):
        for n in g_ricci.nodes():
            g_ricci.node[n]['label'] = 0  # add dummy
    else:  # contains label key
        assert 'label' in list(g_ricci.node[any_node].keys())
        for n in g_ricci.nodes():
            label_norm = 40
            if graph == 'dd_test':
                label_norm = 90
            # label_norm = norm(g, 'label', norm_flag) + 1e-5
            # print(label_norm)
            g_ricci.node[n]['label'] /= float(label_norm)

    if 'deg' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=0)
        if norm_flag == 'yes':
            # better normalization
            for attr in ['1_0_deg_sum']:  # used to include 1_0_deg_std/ deleted now:
                norm_ = norm(g_ricci, attr, norm_flag)
                for n in g_ricci.nodes():
                    g_ricci.node[n][attr] = g_ricci.node[n][attr] / float(norm_)

            # attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=2)
            # attribute_mean(g_ricci, n, key='deg', cutoff=2)
            # attribute_mean(g_ricci, n, key='deg', cutoff=3)
        # for n in g_ricci.nodes():
        #     attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=1)

    if 'label' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='label', cutoff=1, iteration=0)
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='label', cutoff=1, iteration=1)

    if 'cc_min' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='cc')

    if 'ricciCurvature_min' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='ricciCurvature')

    # handle edge filtration case
    if np.array([('edge_' in s) for s in allowed]).all():
        for v in g.nodes():
            assert list(g.nodes[v].keys()) == ['label']
        from .test0 import edge_filtration
        for filtration in allowed:
            assert ((filtration[-3:] == 'ave') or (filtration[-3:] == 'min') or (filtration[-3:] == 'max'))
            if filtration[-3:] == 'ave':
                g_ricci = edge_filtration(g_ricci, function_type=filtration, sub_flag=True, average_flag=True)
            elif filtration[-6:] == 'minmax':
                g_ricci_min = edge_filtration(g_ricci.copy(), function_type=filtration[:-6] + 'min', sub_flag=True,
                                              average_flag=False)
                g_ricci_max = edge_filtration(g_ricci.copy(), function_type=filtration[:-6] + 'max', sub_flag=False,
                                              average_flag=False)
                g_ricci = g_ricci_min
                for v in g_ricci.nodes():
                    assert filtration[:-6] + 'min' in list(g_ricci.node[v].keys())
                    g_ricci.node[v][filtration[:-6] + 'max'] = g_ricci_max.node[v][filtration[:-6] + 'max']
                # for (u,v) in g_ricci.edges(): # not neceesary
                #     g_ricci[u][v][filtration[:-6]+'max'] = g_ricci_max[u][v][filtration[:-6]+'max']

            elif filtration[-3:] == 'min':
                g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type=filtration, average_flag=False)
            elif (filtration[-3:] == 'max') and (filtration[-6:] != 'minmax'):
                g_ricci = edge_filtration(g_ricci, sub_flag=False, function_type=filtration, average_flag=False)
            else:
                print('Unconsidered cases occured in function_basis')
                raise Exception
        # for v in g.node():
        # print 'debug',
        # print g.nodes[v].keys()
        # print (len(g.nodes[v].keys())),
        # print(g.nodes[v].keys()),
        # print(len(allowed))
        # assert len(g.nodes[v].keys()) == (len(allowed) + 1) # 1 here is label

    #
    # if 'edge_p_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, function_type='edge_p_ave', sub_flag=True, average_flag=True)
    # if 'jaccard_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type='jaccard_ave', average_flag=True)
    # if 'jaccard_int_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type='jaccard_int_ave', average_flag=True)
    # if 'ricci_edge_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type='ricci_edge_ave', average_flag=True)
    # if 'ricci_edge_max' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=False, function_type='ricci_edge_max', average_flag=False)
    if False:
        # if transformation_flag==True:

        anynode = list(g_ricci.nodes)[0]
        attribute_lists = list(g_ricci.nodes[anynode].keys())

        for attribute in attribute_lists:
            for v in g_ricci.nodes():
                tmp = g_ricci.nodes[v][attribute]
                g_ricci.nodes[v][attribute] = f_transform(tmp, param={'a': pa, 'b': pb, 'c': 0}, type_='poly')

            # normalization
            # for v in g_ricci.nodes():
            #     g_ricci.nodes[v][attribute] /= norm(g_ricci, key=attribute, flag='yes')
    return g_ricci


def function_basis(g, allowed, norm_flag='no', recomputation_flag=False, transformation_flag=True):
    # function_basis.counter +=1
    # print function_basis.counter,

    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    # allowed = ['ricci', 'deg', 'hop', 'cc', 'fiedler']

    # to save recomputation. Look at the existing feature at first and then simply compute the new one.
    if len(g) < 3:
        return
    existing_features = [list(g.node[list(g.nodes())[0]].keys())]
    if not recomputation_flag:
        allowed = [feature for feature in allowed if feature not in existing_features]
    elif recomputation_flag:
        allowed = allowed
    # print('Recompute only those features', allowed)
    import networkx as nx
    import numpy as np
    assert nx.is_connected(g)
    from GraphRicciCurvature.OllivierRicci import ricciCurvature
    def norm(g, key, flag=norm_flag):
        if flag == 'no':
            return 1
        elif flag == 'yes':
            return np.max(np.abs(list(nx.get_node_attributes(g, key).values()))) + 1e-6
            # return 0.01
            # get the max of g.node[i][key], has some problem actually
            for v, data in sorted(g.node(data=True), key=lambda x: abs(x[1][key]), reverse=True):
                norm = np.float(data[key]) + 1e-6
                return norm

    # ricci
    g_ricci = g
    if 'ricciCurvature' in allowed:
        try:
            g_ricci = ricciCurvature(g, alpha=0.5, weight='weight')
            assert list(g_ricci.node.keys()) == list(g.nodes())
            ricci_norm = norm(g, 'ricciCurvature', norm_flag)
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] /= ricci_norm
        except:
            print('RicciCurvature Error for graph, set 0 for all nodes')
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] = 0
            # don't know what I wrote the follwing code beore
            # ricci_norm = norm(g, 'ricciCurvature', norm_flag)
            # for n in g_ricci.nodes():
            #     g_ricci.node[n]['ricciCurvature'] /= ricci_norm

    # degree
    if 'deg' in allowed:
        deg_dict = dict(nx.degree(g_ricci))
        for n in g_ricci.nodes():
            g_ricci.node[n]['deg'] = deg_dict[n]
            # g_ricci.node[n]['deg'] = np.log(deg_dict[n]+1)

        # deg_norm = np.float(max(deg_dict.values()))
        deg_norm = norm(g_ricci, 'deg', norm_flag)
        for n in g_ricci.nodes():
            g_ricci.node[n]['deg'] /= np.float(deg_norm)

        # return g_ricci

        # hop

    # var
    if 'var' in allowed:
        import scipy
        distance = nx.floyd_warshall_numpy(g);  # return a matrix
        distance = np.array(distance);
        distance = distance.astype(int)
        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            assert n_idx <= len(g_ricci)
            g_ricci.node[n]['var1'] = np.mean(distance[n_idx])
            g_ricci.node[n]['var'] = scipy.stats.moment(distance[n_idx], 2)
            g_ricci.node[n]['var4'] = scipy.stats.moment(distance[n_idx], 4)
        norm_var1 = norm(g_ricci, 'var1', norm_flag)
        norm_var = norm(g_ricci, 'var', norm_flag)
        norm_var4 = norm(g_ricci, 'var4', norm_flag)

        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['var1'] /= np.float(norm_var1)
            g_ricci.node[n]['var'] /= np.float(norm_var)
            g_ricci.node[n]['var4'] /= np.float(norm_var4)

    if 'hop' in allowed:
        distance = nx.floyd_warshall_numpy(g);  # return a matrix
        distance = np.array(distance);
        distance = distance.astype(int)
        if norm_flag == 'no':
            hop_norm = 1
        elif norm_flag == 'yes':
            hop_norm = np.max(distance)
        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            assert n_idx <= len(g_ricci)
            # print(n, n_idx)
            g_ricci.node[n]['hop'] = distance[n_idx][:] / float(hop_norm)

    if 'hop_' in allowed:
        distance = nx.floyd_warshall_numpy(g);  # return a matrix
        distance = np.array(distance);
        distance = distance.astype(int)
        if norm_flag == 'no':
            hop_norm = 1
        elif norm_flag == 'yes':
            hop_norm = np.max(distance)
        for n in g_ricci.nodes():
            # if g_ricci has non consencutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            assert n_idx <= len(g_ricci)
            for n_ in g_ricci.nodes():
                n__idx = list(g_ricci.nodes).index(n_)
                g_ricci.node[n]['hop_' + str(n_)] = distance[n__idx][n_idx] / float(hop_norm)

    # closeness_centrality
    if 'cc' in allowed:
        closeness_centrality = nx.closeness_centrality(g)  # dict
        closeness_centrality = {k: v / min(closeness_centrality.values()) for k, v in
                                closeness_centrality.items()}  # no normalization for debug use
        closeness_centrality = {k: 1.0 / v for k, v in closeness_centrality.items()}
        for n in g_ricci.nodes():
            g_ricci.node[n]['cc'] = closeness_centrality[n]

    # fiedler
    if 'fiedler' in allowed:
        from networkx.linalg.algebraicconnectivity import fiedler_vector
        fiedler = fiedler_vector(g, normalized=False)  # np.ndarray
        assert max(fiedler) > 0
        fiedler = fiedler / max(np.abs(fiedler))
        assert max(np.abs(fiedler)) == 1
        for n in g_ricci.nodes():
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['fiedler'] = fiedler[n_idx]

    if 'e1' in allowed:
        import scipy
        eigvs = scipy.linalg.eig(nx.laplacian_matrix(g).todense())[1]
        if len(eigvs) <= 6:
            n = len(eigvs)
            first3 = np.array([[1] * n] * 3)
            last3 = np.array([[1] * n] * 3)
        else:
            first3 = eigvs[0:3]
            last3 = eigvs[-3:]
        for n in g_ricci.nodes():
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['e1'] = first3[0][n_idx] / max(abs(first3[0]))
            g_ricci.node[n]['e2'] = first3[1][n_idx] / max(abs(first3[1]))
            g_ricci.node[n]['e3'] = first3[2][n_idx] / max(abs(first3[2]))
            g_ricci.node[n]['e-1'] = last3[0][n_idx] / max(abs(last3[0]))
            g_ricci.node[n]['e-2'] = last3[1][n_idx] / max(abs(last3[1]))
            g_ricci.node[n]['e-3'] = last3[2][n_idx] / max(abs(last3[2]))

    any_node = list(g_ricci.node)[0]
    if 'label' not in list(g_ricci.node[any_node].keys()):
        for n in g_ricci.nodes():
            g_ricci.node[n]['label'] = 0  # add dummy
    else:  # contains label key
        assert 'label' in list(g_ricci.node[any_node].keys())
        for n in g_ricci.nodes():
            label_norm = 40
            if graph == 'dd_test':
                label_norm = 90
            # label_norm = norm(g, 'label', norm_flag) + 1e-5
            # print(label_norm)
            g_ricci.node[n]['label'] /= float(label_norm)

    if 'deg' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=0)
        if norm_flag == 'yes':
            # better normalization
            for attr in ['1_0_deg_sum']:  # used to include 1_0_deg_std/ deleted now:
                norm_ = norm(g_ricci, attr, norm_flag)
                for n in g_ricci.nodes():
                    g_ricci.node[n][attr] = g_ricci.node[n][attr] / float(norm_)

            # attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=2)
            # attribute_mean(g_ricci, n, key='deg', cutoff=2)
            # attribute_mean(g_ricci, n, key='deg', cutoff=3)
        # for n in g_ricci.nodes():
        #     attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=1)

    if 'label' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='label', cutoff=1, iteration=0)
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='label', cutoff=1, iteration=1)

    if 'cc_min' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='cc')

    if 'ricciCurvature_min' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='ricciCurvature')

    # handle edge filtration case
    if np.array([('edge_' in s) for s in allowed]).all():
        for v in g.nodes():
            assert list(g.nodes[v].keys()) == ['label']
        from .test0 import edge_filtration
        for filtration in allowed:
            assert ((filtration[-3:] == 'ave') or (filtration[-3:] == 'min') or (filtration[-3:] == 'max'))
            if filtration[-3:] == 'ave':
                g_ricci = edge_filtration(g_ricci, function_type=filtration, sub_flag=True, average_flag=True)
            elif filtration[-6:] == 'minmax':
                g_ricci_min = edge_filtration(g_ricci.copy(), function_type=filtration[:-6] + 'min', sub_flag=True,
                                              average_flag=False)
                g_ricci_max = edge_filtration(g_ricci.copy(), function_type=filtration[:-6] + 'max', sub_flag=False,
                                              average_flag=False)
                g_ricci = g_ricci_min
                for v in g_ricci.nodes():
                    assert filtration[:-6] + 'min' in list(g_ricci.node[v].keys())
                    g_ricci.node[v][filtration[:-6] + 'max'] = g_ricci_max.node[v][filtration[:-6] + 'max']
                # for (u,v) in g_ricci.edges(): # not neceesary
                #     g_ricci[u][v][filtration[:-6]+'max'] = g_ricci_max[u][v][filtration[:-6]+'max']

            elif filtration[-3:] == 'min':
                g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type=filtration, average_flag=False)
            elif (filtration[-3:] == 'max') and (filtration[-6:] != 'minmax'):
                g_ricci = edge_filtration(g_ricci, sub_flag=False, function_type=filtration, average_flag=False)
            else:
                print('Unconsidered cases occured in function_basis')
                raise Exception
        # for v in g.node():
        # print 'debug',
        # print g.nodes[v].keys()
        # print (len(g.nodes[v].keys())),
        # print(g.nodes[v].keys()),
        # print(len(allowed))
        # assert len(g.nodes[v].keys()) == (len(allowed) + 1) # 1 here is label

    #
    # if 'edge_p_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, function_type='edge_p_ave', sub_flag=True, average_flag=True)
    # if 'jaccard_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type='jaccard_ave', average_flag=True)
    # if 'jaccard_int_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type='jaccard_int_ave', average_flag=True)
    # if 'ricci_edge_ave' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=True, function_type='ricci_edge_ave', average_flag=True)
    # if 'ricci_edge_max' in allowed:
    #     g_ricci = edge_filtration(g_ricci, sub_flag=False, function_type='ricci_edge_max', average_flag=False)
    if False:
        # if transformation_flag==True:

        anynode = list(g_ricci.nodes)[0]
        attribute_lists = list(g_ricci.nodes[anynode].keys())

        for attribute in attribute_lists:
            for v in g_ricci.nodes():
                tmp = g_ricci.nodes[v][attribute]
                g_ricci.nodes[v][attribute] = f_transform(tmp, param={'a': pa, 'b': pb, 'c': 0}, type_='poly')

            # normalization
            # for v in g_ricci.nodes():
            #     g_ricci.nodes[v][attribute] /= norm(g_ricci, key=attribute, flag='yes')
    return g_ricci


def add_function_value(gi, fv_input='fv_test', edge_value='max'):
    # gi is nx graph with cc, deg, fiedler, hop, lebael, ricci computed
    # add fv function value for edges
    # fv_input here is use the existing fv as fv

    import random
    # deg = dict(gi.degree())
    # closeness_centrality_dict = nx.closeness_centrality(gi)
    # for n in closeness_centrality_dict.keys():
    #     closeness_centrality_dict[n] = 1 / closeness_centrality_dict[n]

    fv_test = {i: gi.nodes[i]['fv_test'] for i in gi.nodes()}  # the key is not necessarily consecutive

    # legacy code
    if fv_input == 'deg':
        pass
        # fv = deg
    elif fv_input == 'closeness_centrality':
        pass
        # fv = closeness_centrality_dict
    elif fv_input == 'fv_test':
        fv = fv_test

    assert type(fv) == dict
    # fv_list.sort()
    fv_random = {}
    for i in list(fv.keys()):
        fv_random[i] = fv[i] + random.uniform(0, 1e-8)
    # assert len(np.unique(fv_list)) == len(fv_list)
    # needs to consider non-consecutive case

    for i in gi.node():
        # gi.node[i]['deg'] = deg[i]
        # gi.node[i]['closeness_centrality'] = closeness_centrality_dict[i]
        gi.node[i]['fv'] = fv[i]
        gi.node[i]['fv_random'] = fv_random[i]

    for (e1, e2) in gi.edges():
        if edge_value == 'max':
            gi[e1][e2]['fv'] = max(gi.node[e1]['fv'], gi.node[e2]['fv'])
            gi[e1][e2]['fv_random'] = max(gi.node[e1]['fv_random'], gi.node[e2]['fv_random'])
        if edge_value == 'min':
            gi[e1][e2]['fv'] = min(gi.node[e1]['fv'], gi.node[e2]['fv'])
            gi[e1][e2]['fv_random'] = min(gi.node[e1]['fv_random'], gi.node[e2]['fv_random'])
        assert type(fv_random) == dict
    tmp = list(fv_random.values())
    tmp.sort()
    return (gi, tmp)
    # return (gi, fv_list)


@timefunction
def compute_graphs_(allowed, graph_isomorphisim, graph, norm_flag='no', feature_addition_flag=False,
                    skip_dump_flag='yes'):
    (graphs_tmp, message) = load_data(graph, 'dgms_normflag_' + norm_flag, beta=-1, no_load='yes')
    if message == 'success':
        return graphs_tmp

    if graph_isomorphisim == 'off':
        (graphs_tmp, flag) = load_data(graph, 'graphs_', no_load='yes')
        if flag != 'success':
            from joblib import Parallel, delayed
            graphs_tmp = Parallel(n_jobs=-1, batch_size='auto')(
                delayed(pipeline0)(i, allowed, norm_flag=norm_flag, feature_addition_flag=feature_addition_flag) for i
                in range(len(data)))
            # assert 'cc' in graphs_tmp[0][0].node[0].keys()
            # assert 'deg' in graphs_tmp[0][0].node[0].keys()
            # assert 'cc' not in graphs_[0][0].node[0].keys()
            # assert 'deg' in graphs_[0][0].node[0].keys()
            dump_data(graph, graphs_tmp, 'graphs_', skip=skip_dump_flag)
        print()
    elif graph_isomorphisim == 'on':
        nx_graphs_ = generate_graphs()[100:]
        size = 8
        nx_graphs_ = read_all_graphs(size)
        import random
        nx_graphs_ = [nx_graphs_[i] for i in sorted(random.sample(range(len(nx_graphs_)), 2000))]
        graphs_tmp = Parallel(n_jobs=n_jobs, batch_size='auto')(
            delayed(pipeline0)(i, allowed, version=2, norm_flag=norm_flag) for i in range(len(nx_graphs_)))
        print(('Total number of non isomorphic simple graphs of size %s is %s' % (size, len(graphs_))))
    dump_data(graph, graphs_tmp, 'dgms_normflag_' + norm_flag, beta=-1, still_dump='yes', skip='yes')

    if (graph == 'imdb_binary') or (graph == 'imdb_multi') or (graph == 'dd_test') or (graph == 'protein_data') or (
            graph == 'collab'):
        uniform_norm_flag = True
    else:
        uniform_norm_flag = False
    if norm_flag == 'yes': uniform_norm_flag = False
    if uniform_norm_flag:
        anynode = list(graphs_tmp[0][0].nodes)[0]
        print((graphs_tmp[0][0].nodes[anynode]))
        attribute_lists = list(graphs_tmp[0][0].nodes[anynode].keys())
        attribute_lists = [attribute for attribute in attribute_lists if attribute != 'hop']
        for attribute in attribute_lists:
            max_ = 0;
            tmp_max_ = []
            min_ = 1;
            tmp_min_ = []
            for i in range(len(graphs_tmp)):
                if len(graphs_tmp[i]) == 0:
                    print(('skip graph %s' % i))
                    continue
                tmp_max_ += [
                    np.max(list(nx.get_node_attributes(graphs_tmp[i][0], attribute).values()))]  # catch exception
                # except:
                #     print (i, graphs_tmp[i])
                tmp_min_ += [np.min(list(nx.get_node_attributes(graphs_tmp[i][0], attribute).values()))]
            from heapq import nlargest
            print((nlargest(5, tmp_min_)[-1]))
            denominator = max(nlargest(10, tmp_max_)[-1], nlargest(10, np.abs(tmp_min_))[-1]) + 1e-10
            # denominator = max(max(np.abs(tmp_max_)),max(np.abs(tmp_min_)))+1e-10
            print(('Attribute and demoninator: ', attribute, denominator))

            for i in range(len(graphs_tmp)):
                if len(graphs_tmp[i]) == 0:
                    continue
                n_component = len(graphs_tmp[i])
                for comp in range(n_component):
                    for v in graphs_tmp[i][comp].nodes():
                        graphs_tmp[i][comp].nodes[v][attribute] = graphs_tmp[i][comp].nodes[v][attribute] / np.float(
                            denominator)
                        # graphs_tmp[i][comp].nodes[v][attribute + '_uniform'] = graphs_tmp[i][comp].nodes[v][attribute]/ np.float(denominator)
                        # print graphs_tmp[i][0].nodes[v][attribute],

    return graphs_tmp
    print('Finish pipeline 0(Convert to nx, calculate deg, ricci, hop...)')


@timefunction
def kernelsvm(dist_matrix, Y, c, sigma, dist_flag='yes', print_flag='off'):
    import numpy as np
    from joblib import delayed, Parallel
    if dist_flag == 'yes':
        kernel = np.exp(np.multiply(dist_matrix, -dist_matrix) / sigma)
    elif dist_flag == 'no':
        kernel = dist_matrix
    svm_data = Parallel(n_jobs=-1)(
        delayed(single_test)(kernel, c, Y, r_seed, lbd=1e-7, loss=loss_type, debug='off') for r_seed in range(10))
    accuracies = get_accuracies(svm_data, c, sigma, threshold, loss=loss_type)
    print(('svm kernel result: c: %s, sigma %s, train_acc: %s, test_acc: %s' % (
    c, sigma, accuracies['train_acc'], accuracies['test_acc'])))

    if print_flag == 'on':
        print((c, sigma, accuracies['train_acc'], accuracies['test_acc']))
        print((c, sigma, accuracies['train_accuracies'][0], accuracies['test_accuracies']))
    return (kernel, svm_data)


@timefunction
def searchclf(X, Y, i, test_size=0.1, nonlinear_flag='False', verbose=0, print_flag='off', laplacian_flag=False,
              lap_band=10):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn import svm
    from sklearn.metrics import classification_report
    from sklearn.metrics.pairwise import laplacian_kernel

    if nonlinear_flag == 'True':
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 1000]},
                            {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10, 100], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    elif nonlinear_flag == 'False':
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    elif nonlinear_flag == 'reddit_12K':
        # return {'kernel': 'linear', 'C': 1000}
        tuned_parameters = [{'kernel': ['linear'], 'C': [1000]},
                            {'kernel': ['rbf'], 'gamma': [1, 10, 100], 'C': [1000]}]

    if laplacian_flag == True:  # test RetGK idea, will overwrite tuned parameters
        tuned_parameters = [{'kernel': ['precomputed'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=i)
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=verbose)
        if laplacian_flag == False:
            clf.fit(X_train, y_train)
        elif laplacian_flag == True:
            kernel_train = laplacian_kernel(X_train, X_train, gamma=lap_band)
            clf.fit(kernel_train, y_train)

        if print_flag == 'on':
            print(("Best parameters set found on development set is \n %s with score %s" % (
                clf.best_params_, clf.best_score_)))
            print((clf.best_params_))
            print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        if print_flag == 'on':
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print(("%0.3f (+/-%0.03f) for %r"
                       % (mean, std * 2, params)))
            print("Detailed classification report:\n")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            if laplacian_flag == False:
                y_true, y_pred = y_test, clf.predict(X_test)
            elif laplacian_flag == True:
                kernel_test = laplacian_kernel(X_test, X_train, gamma=1)
                y_true, y_pred = y_test, clf.predict(kernel_test)
            print((classification_report(y_true, y_pred)))

    if laplacian_flag == True:
        best_params_ = clf.best_params_
        best_params_['gamma'] = lap_band
        return best_params_
    else:
        return clf.best_params_


@timefunction
def evaluate_clf(graph, X, Y, best_params_, n_splits):
    from sklearn import svm
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + 'neuron' + '/result/'
    accuracy = []
    if (graph == 'reddit_5K'):
        n = 5
    elif graph == ('reddit_12K'):
        n = 2
    else:
        n = 5
    for i in range(n):
        # after grid search, the best parameter is {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
        if best_params_['kernel'] == 'linear':
            clf = svm.SVC(kernel='linear', C=best_params_['C'])
        elif best_params_['kernel'] == 'rbf':
            clf = svm.SVC(kernel='rbf', C=best_params_['C'], gamma=best_params_['gamma'])
        elif best_params_['kernel'] == 'precomputed':  # take care of laplacian case
            clf = svm.SVC(kernel='precomputed', C=best_params_['C'])
        else:
            raise Exception('Parameter Error')

        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import StratifiedKFold
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)

        if clf.kernel == 'precomputed':
            from sklearn.metrics.pairwise import laplacian_kernel
            laplacekernel = laplacian_kernel(X, X, gamma=best_params_['gamma'])
            cvs = cross_val_score(clf, laplacekernel, Y, n_jobs=-1, cv=k_fold)
            print('CV Laplacian kernel')
        else:
            cvs = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold)

        print(cvs)
        acc = cvs.mean()
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    print(('mean is %s, std is %s ' % (accuracy.mean(), accuracy.std())))
    return (accuracy.mean(), accuracy.std())


def baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=False, skip_rf=False, laplacian_flag=False, lap_band=10,
             skip_svm=False):
    import time
    if not skip_rf:
        rfresult = rfclf(X, Y, m_f='auto', multi_cv_flag=multi_cv_flag)
    elif skip_rf:
        rfresult = [0, 0]
    rfresult = ["{0:.1f}".format(100 * i) for i in rfresult]

    if skip_svm:
        return (rfresult[0], rfresult[1], '0', '0', '0', '0')

    time1 = time.time()
    print('Using deg function as a baseline')
    if graph == 'reddit_12K':
        nonlinear_flag = 'reddit_12K'
        param = searchclf(X, Y, 1001, test_size=0.1, nonlinear_flag=nonlinear_flag, verbose=0,
                          laplacian_flag=laplacian_flag, lap_band=lap_band, print_flag='on')
        # param = {'kernel': 'linear', 'C': 1000}
    else:
        nonlinear_flag = 'True'
        param = searchclf(X, Y, 1001, test_size=0.1, nonlinear_flag=nonlinear_flag, verbose=0,
                          laplacian_flag=laplacian_flag, lap_band=lap_band)
    svm_result = evaluate_clf(graph, X, Y, param, n_splits=n_splits)
    svm_result = ["{0:.1f}".format(100 * i) for i in svm_result]
    time2 = time.time()

    return (rfresult[0], rfresult[1], svm_result[0], svm_result[1], round(time2 - time1), str(param))


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        print(("values: {}".format(values)))
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def readargs():
    import sys
    import numpy as np
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', help='The type of graph', default='mutag')
    parser.add_argument('-rf', '--rf_flag', help='y if only use rf', default='n')
    parser.add_argument('-b', '--beta', help='The combination of deg, ricci, fiedler, cc and label',
                        default=np.array([1, 1, 0, 0, 0]), type=np.ndarray)
    parser.add_argument('-norm_flag', '--norm_flag',
                        help='yes means normalize when computing function, no means no normalization(default)',
                        default='no')
    parser.add_argument('-high_order_flag', '--high_order_flag', help='Decide whether to use high order graphs',
                        default=False)
    parser.add_argument('-p', '--parameter', help='The parameter of filtration function',
                        default={'a': 0, 'b': 1, 'c': 0}, type=json.loads)
    parser.add_argument('-pa', '--pa', help='The parameter of filtration function', default=0, type=float)
    parser.add_argument('-pb', '--pb', help='The parameter of filtration function', default=1, type=float)

    # parser.add_argument('-kp', "--keypairs", dest="my_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL")
    args = parser.parse_args()

    if len(sys.argv) != 1:
        graph = args.graph
        beta = args.beta
        rf_flag = args.rf_flag
        norm_flag = args.norm_flag
        high_order_flag = args.high_order_flag
        parameter_flag = args.parameter
        pa = args.pa;
        pb = args.pb
    else:
        graph = 'mutag'
        beta = np.array([1, 1, 0, 0, 0])
        beta = beta / np.float(np.sum(beta))
        parameter_flag = {'a': 0, 'b': 1, 'c': 0}
        # keypairs = {'a':0, 'b':1, 'c':0}
        pa = 0;
        pb = 1;
        assert abs(np.sum(beta) - 1) < 0.01
        print(('Using default parameters, data is %s, beta is %s' % (graph, beta)))
    return (graph, rf_flag, beta, high_order_flag, parameter_flag, pa, pb)


def get_i_dist_distribution(i, print_flag=False, cdf_flag=False):
    import time
    t1 = time.time()
    gs = graphs_[i]
    # gs = globals()[str(graphs_) + str(i)] # test for memory
    result = np.zeros((1, 30))
    for g in gs:
        result += dist_distribution(g, cdf_flag=cdf_flag)
    print(('-'), end=' ')
    try:
        if print_flag:
            print((i, len(gs[0]), len(gs[0].edges()), time.time() - t1))
    except:
        print(('Graphs %s has some problem' % i))
    return result


def test():
    for i in range(1, 5000, 100):
        import time
        t1 = time.time()
        get_i_dist_distribution(1)
        if time.time() - t1 > 5:
            print(('Graph of size %s %i takes %s' % (i, graphs_[i][0], time.time() - t1)))

    # g = graphs_[0][0]
    # np.shape(dist_distribution(g))
    # get_i_dist_distribution(12)


def cached(cachefile):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):  # define a decorator for a function "fn"

        def wrapped(*args, **kwargs):  # define a wrapper that will finally call "fn" with all arguments
            # if cache exists -> load it and return its content
            import pickle, os
            global graph
            direct = set_baseline_directory(graph)
            if os.path.exists(direct + cachefile):
                with open(direct + cachefile, 'rb') as cachehandle:
                    print((" using cached result from '%s'" % direct + cachefile))
                    return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(direct + cachefile, 'wb') as cachehandle:
                print(("saving result to cache '%s'" % cachefile))
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator  # return this "customized" decorator that uses "cachefile"


# @cached('distance_distribution')
def get_dist_distribution(n_jobs=-1, print_flag=False, cdf_flag=False):
    n = len(graphs_)
    from joblib import Parallel, delayed
    total = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(get_i_dist_distribution)(i, print_flag=print_flag, cdf_flag=cdf_flag) for i in range(n))
    total = np.array(total)
    feature = np.stack(total, axis=1)[0]
    return feature


def handle_edge_filtration(edge_fil):
    if edge_fil[-6:] != 'minmax':
        assert edge_fil[-3:] == 'ave'
        (graphs, dgms, sub_dgms, super_dgms, epd_dgms) = dgms_data(graph, np.array(beta), n_jobs, debug_flag,
                                                                   hop_flag='n', basep=0, edge_fil=edge_fil)

    # handle minmax case. Need to merge/throw half of dgms
    if edge_fil[-6:] == 'minmax':
        edge_fil_min = edge_fil[:-6] + 'min'
        (min_graphs, min_dgms, min_sub_dgms, min_super_dgms, min_epd_dgms) = dgms_data(graph, np.array(beta), n_jobs,
                                                                                       debug_flag, hop_flag='n',
                                                                                       basep=0,
                                                                                       edge_fil=edge_fil_min)
        edge_fil_max = edge_fil[:-6] + 'max'
        (max_graphs, max_dgms, max_sub_dgms, max_super_dgms, max_epd_dgms) = dgms_data(graph, np.array(beta), n_jobs,
                                                                                       debug_flag, hop_flag='n',
                                                                                       basep=0,
                                                                                       edge_fil=edge_fil_max)
        sub_dgms = min_sub_dgms;
        super_dgms = max_super_dgms;
        dgms = [add_dgms(sub_dgms[i], super_dgms[i]) for i in range(len(sub_dgms))]
        epd_dgms = [add_dgms(min_epd_dgms[i], max_epd_dgms[i]) for i in range(len(sub_dgms))]  # quick hack
    for dgm in dgms:
        print_dgm(dgm)
    return (sub_dgms, super_dgms, dgms, epd_dgms)


def test_tomorrow():
    import community
    import matplotlib.pyplot as plt
    print((nx.info(g)))
    parts = community.best_partition(g)
    type(parts)
    list(parts.keys())
    values = [parts.get(node) for node in g.nodes()]
    len(values)
    plt.axis("off")
    nx.draw_networkx(g, cmap=plt.get_cmap("jet"), node_color=values, node_size=35, with_labels=False)
    nx.show()


def set_table(graph, hyperharameter):
    table = PrettyTable();
    n_row = 5
    (hyperparameter_flag, norm_flag, loss_type, n_runs, pd_flag, multi_cv_flag, n_jobs, debug_flag, graph_isomorphisim,
     edge_fil, dynamic_range_flag,) = hyperparameter
    space = '  '

    title = 'hyperparameter_flag = ' + str(hyperparameter_flag) + space + \
            '  norm_flag = ' + str(norm_flag) + space + \
            '  loss_type = ' + str(loss_type) + space + \
            '  n_runs = ' + str(n_runs) + space + \
            '  multi_cv_flag = ' + str(multi_cv_flag) + space + \
            '  n_jobs = ' + str(n_jobs) + space + \
            '  debug_flag =' + str(debug_flag) + space + \
            '  graph_isomorphisim =' + str(graph_isomorphisim) + space + \
            '  edge_fil = ' + str(edge_fil) + space + \
            '  dynamic_range_flag = ' + str(dynamic_range_flag) + '\n'
    print(title)
    table.field_names = [graph, "DT", 'RF', "SVM/Std", 'Time/Kernel Param/PD Stat'];
    return (table, n_row)


def set_hyperparameter():
    hyperparameter_flag = 'no'
    norm_flag = 'no'
    loss_type = 'hinge'
    n_runs = 10
    pd_flag = 'True'
    multi_cv_flag = True
    n_jobs = -1
    debug_flag = 'off'
    graph_isomorphisim = 'off'
    edge_fil = 'off'
    dynamic_range_flag = True
    return (
    hyperparameter_flag, norm_flag, loss_type, n_runs, pd_flag, multi_cv_flag, n_jobs, debug_flag, graph_isomorphisim,
    edge_fil, dynamic_range_flag)


def edge_baseline(edge_fil, graph, table):
    global data, Y, graphs_, X, Y, n
    graphs_ = compute_graphs_([edge_fil], graph_isomorphisim, graph, norm_flag=norm_flag)

    # edge baseline
    X = merge_features(graphs_, [edge_fil], 30, his_norm_flag='yes', edge_flag=True)
    print('Without distance feature')
    for ax in [0, 1]:
        X = normalize_(X, axis=ax);
        bl_result = baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag)
        table.add_row(['Edge BL ' + str(ax) + str(' ') + str(edge_fil), bl_result[0],
                       bl_result[1], bl_result[2] + '/' + bl_result[3],
                       ' time: ' + str(round(bl_result[4])) + str(' ') + bl_result[5]])
        print(table)

        # check edge attribute
        print((list(graphs_[0][0].node[0].keys())))
        global beta
        beta = unit_vector(5, 0)  # dummy
        (sub_dgms, super_dgms, dgms) = handle_edge_filtration(edge_fil)
        for i in range(len(dgms)):
            export_dgm(graph, dgms[i], i, filename=edge_fil, flag='ss')
            export_dgm(graph, sub_dgms[i], i, filename=edge_fil, flag='sub')
            export_dgm(graph, super_dgms[i], i, filename=edge_fil, flag='super')
        print('Finish exporting edge based dgms')
        continue
        print(('Using edge filtration %s' % edge_fil))
        (rf_, cv_score, t, svm_param, dgms_stat1, dgms_stat2, vct) = clf_pdvector(0, (sub_dgms, super_dgms, dgms), beta,
                                                                                  Y, pd_flag, print_flag='off',
                                                                                  nonlinear_flag=True, axis=axis,
                                                                                  rf_flag=rf_flag,
                                                                                  dynamic_range_flag=dynamic_range_flag)  # use pd vector as baseline
        table.add_row([edge_fil + ' pd vector ax=' + str(ax), rf_[0], rf_[1], cv_score[0] + '/' + str(cv_score[1]),
                       'vct:' + vct + ' /svm time: ' + t + ' ' + svm_param + str(dgms_stat1)])

        for kernel_type in ['sw', 'pss', 'wg']:
            tda_kernel_data = (0, 0, {})
            for bandwidth in [0.1, 1, 10, 100]:
                (tda_kernel, t1) = sw_parallel(dgms2swdgm(dgms), dgms2swdgm(dgms), parallel_flag=True,
                                               kernel_type=kernel_type, n_directions=10, bandwidth=bandwidth)
                tda_kernel_data_ = evaluate_tda_kernel(tda_kernel, Y)
                if tda_kernel_data_[0] > tda_kernel_data[0]:
                    tda_kernel_data = tda_kernel_data_  # only print best one
            table.add_row([edge_fil + ' ' + str(kernel_type) + ' bw:' + str(bandwidth), '', '',
                           str(tda_kernel_data[0]) + '/' + str(tda_kernel_data[1]),
                           'kct: ' + str(t1) + '/svm_time: ' + str(tda_kernel_data[3]) + str(tda_kernel_data[2])])
            print(table)

    del X, beta
    return table


def basepoints_aggregation(graph, n, flag='off'):
    if flag == 'off':
        return
    global X
    X = basept_vector(graph, n, normalize_ax=1, upper_bound=45)
    X = np.delete(X, np.where(~X.any(axis=0))[0], axis=1)
    rfclf(X, Y, m_f=20)
    param = searchclf(X, Y, 1001, test_size=0.1, nonlinear_flag='False', verbose=1)
    evaluate_clf(graph, X, Y, param, n_splits=10)


# @cached('edge_betweenness')
def edge_btwn_features(upperbound=0.4, pro_flag=False, dummy_flag=False, n_jobs=20, print_flag=False):
    n = len(graphs_)
    if dummy_flag:
        return np.zeros((n, 1))
    from joblib import delayed, Parallel
    result = Parallel(n_jobs=n_jobs, batch_size='auto', verbose=5)(
        delayed(edge_btwn_i_feature)(i, upperbound, pro_flag=pro_flag, print_flag=print_flag) for i in range(n))
    result = np.vstack(result)
    result = np.delete(result, np.where(~result.any(axis=0))[0], axis=1)
    # result = normalize_(result, axis=axis)
    return result


def edge_btwn_i_feature(i, upperbound=0.4, pro_flag=False, print_flag=False):
    import time
    t1 = time.time()
    betweenness_upperbound = upperbound
    edge_btwn_list = []
    for g in graphs_[i]:
        if g == None:
            continue
        edge_btwn_list += list(nx.edge_betweenness_centrality(g).values())
    result = np.histogram(edge_btwn_list, bins=300, range=(0, betweenness_upperbound))[0]  # initial range=(0,0.4)
    if pro_flag:  # encode the connectivity of edge feature
        result = list(result) + edge_btwn_i_feature_(i, upperbound=upperbound)
    print(('-'), end=' ')
    if print_flag == print_flag:
        try:
            print((i, len(graphs_[i][0]), time.time() - t1))
        except IndexError:  # for some i, graphs_ is empty/ very rare case
            print(('Graph_[%s] has problem(is empty)' % i))
    return np.array(result)


def edge_btwn_i_feature_(i, upperbound=0.4):
    global graphs_
    final_result = [0] * 120
    for j in range(len(graphs_[i])):
        g = graphs_[i][j]
        btwn_dict = nx.edge_betweenness_centrality(g)
        for e in g.edges():
            (a, b) = e
            val_list = get_nbr_edge_vals(g, (a, b), btwn_dict)
            stat = list_stat(val_list)
            for key in ['mean', 'min', 'max', 'std']:
                g[a][b][key] = stat[key]

        result = []
        for key in ['mean', 'min', 'max', 'std']:
            hisresult = list(nx.get_edge_attributes(g, key).values())
            result += list(np.histogram(hisresult, bins=30, range=(0, upperbound))[0])
        final_result = list(np.array(final_result) + np.array(result))
    try:
        assert sum(result) == sum([len(g.edges) for g in graphs_[i]]) * 4
    except AssertionError:
        print((i, sum(result), sum([len(g.edges) for g in graphs_[i]]) * 4))
    return final_result


def edge_feature(i, attr='deg', plus_flag=True):
    g = graphs_[i][0]
    feature_list = []
    for edge in g.edges():
        node1 = edge[0]
        node2 = edge[1]
        dict1 = g.node[node1]
        dict2 = g.node[node2]
        if plus_flag == True:
            feature_list.append(0.5 * (dict1[attr] + dict2[attr]))
        elif plus_flag == False:
            feature_list.append(np.abs(0.5 * (dict1[attr] - dict2[attr])))
    return feature_list


def edge_histogram(graphs_, attributes=['deg']):
    edge_hist = np.zeros((len(graphs_), 1))
    for attr in attributes:
        for plus_flag in [True, False]:
            edge_hist_tmp = np.zeros((len(graphs_), 200))
            n = len(graphs_)
            for i in range(n):
                edge_hist_tmp[i] = hisgram(edge_feature(i, attr, plus_flag=plus_flag), n_bin=200)
            print((attr, plus_flag))
            edge_hist_tmp = remove_zero_col(edge_hist_tmp)
            print((np.shape(edge_hist_tmp)))
            edge_hist = np.concatenate((edge_hist, edge_hist_tmp), axis=1)
            print((np.shape(edge_hist)))
    return remove_zero_col(edge_hist)


def laplacian_str(laplacian_flag=False, lap_bandwidth=10):
    if laplacian_flag == False:
        return ''
    elif laplacian_flag == True:
        return 'LapBL ' + str(lap_bandwidth) + ' '


def node_baseline(graph, Y, graphs_, table, allowed, bin_size=50, extra_feature=[],
                  norm_flag_='yes', cdf_flag=False, edge_structure_flag=False, high_order_flag=False,
                  coarse_flag=True, edge_feature_flag=False, pro_flag=False, skip_flag=False, print_flag=False,
                  laplacian_flag=False, lap_bandwidth=10, uniform_flag=True, sanity_flag=False):
    # graphs_ = compute_graphs_(table, allowed, graph, norm_flag=norm_flag_)
    assert set(allowed).issubset(set(graphs_[0][0].node[0].keys()))
    if edge_feature_flag:
        distance_feature_orgin = get_dist_distribution(n_jobs=-1, print_flag=print_flag, cdf_flag=cdf_flag)
        btwn_feature_origin = edge_btwn_features(upperbound=0.5, pro_flag=pro_flag, n_jobs=-1, dummy_flag=False,
                                                 print_flag=print_flag)

    baseline_feature = ['1_0_deg_min', '1_0_deg_max', '1_0_deg_mean', 'deg']
    X_origin = merge_features(graph, graphs_, baseline_feature + extra_feature, bin_size, his_norm_flag=norm_flag_,
                              cdf_flag=cdf_flag, uniform_flag=uniform_flag)

    for ax in [0, 1]:
        if (graph == 'reddit_12K') and (
                ax == 0):  # notice that ax 0 is useless for reddit 12k when using linear svm with c = 1000
            continue
        if high_order_flag == 'True':
            from joblib import delayed, Parallel
            high_order_graphs = Parallel(n_jobs=-1)(delayed(high_order)(graphs_[i][0]) for i in range(len(graphs_)))

            # high_order_graphs = [high_order(graphs_[i][0]) for i in range(len(graphs_))]

            high_order_features_2 = high_order_feature(n, high_order_graphs, axis=ax, order=2)
            high_order_features_3 = high_order_feature(n, high_order_graphs, axis=ax, order=3)
            high_order_features_4 = high_order_feature(n, high_order_graphs, axis=ax, order=4)
            high_order_features = np.concatenate((high_order_features_2, high_order_features_3, high_order_features_4),
                                                 axis=1)
            X_origin = normalize_(X_origin, axis=ax)
            bl_result = baseline(X_origin, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag,
                                 laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)

            print((X_origin, high_order_features))
            X_origin = np.concatenate((X_origin, high_order_features), axis=1)
            print((np.shape(X_origin)))
            bl_result = baseline(X_origin, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag,
                                 laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)
            print(bl_result)

        if not skip_flag:
            print(('Baseline Model, normalize %s axis' % ax))
            print('Without distance feature')
            X = normalize_(X_origin, axis=ax)

            if coarse_flag:
                coarseX = merge_features(graph, coarsed_graphs_, baseline_feature + extra_feature, 30,
                                         his_norm_flag=norm_flag_, uniform_flag=uniform_flag)
                coarseX = normalize_(coarseX, axis=ax)
                X = np.append(X, coarseX, axis=1)
            if sanity_flag == True:
                n_graph = len(graphs_)
                extra_feature_ = np.zeros((n_graph, 2))
                for i in range(n_graph):
                    if len(graphs_[i]) > 0:
                        extra_feature_[i][0] = len(graphs_[i][0])
                        extra_feature_[i][1] = nx.number_of_edges(graphs_[i][0])
                        # extra_feature_[i][2] = nx.diameter(graphs_[i][0])
                    if len(graphs_[i]) > 1:
                        pass
                        # extra_feature_[i][3] = len(graphs_[i][1])
                        # extra_feature_[i][4] = nx.number_of_edges(graphs_[i][1])
                        # extra_feature_[i][5] = nx.diameter(graphs_[i][1])
                extra_feature_ = normalize_(extra_feature_, axis=ax)
                print(extra_feature_)
                X = np.concatenate((X, extra_feature_), axis=1)

            bl_result = baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag,
                                 laplacian_flag=laplacian_flag, lap_band=lap_bandwidth, skip_rf=False)
            print('------------------------------------------')
            lapstr = laplacian_str(laplacian_flag, lap_bandwidth)
            table.add_row(
                [lapstr + 'Bin: ' + str(bin_size) + ' Node BL(deg) ' + str(ax) + str(' ') + str(norm_flag_) + str(
                    extra_feature), bl_result[0],
                 bl_result[1], bl_result[2] + '/' + bl_result[3],
                 ' time:' + str(bl_result[4]) + str(' ') + bl_result[5]])
            print(table)

        if edge_structure_flag:
            X = normalize_(X_origin, axis=ax)
            print('With Structure feature')
            structure_feature = normalize_(edge_histogram(graphs_, attributes=['deg', '1_0_deg_mean', '1_0_deg_sum']),
                                           axis=ax)
            X = np.concatenate((X, structure_feature), axis=1)
            X = normalize_(X, axis=ax);
            bl_result = baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag,
                                 laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)
            table.add_row(
                ['Node BL(deg)+Structure Feature ' + str(ax) + str(' ') + str(norm_flag_) + str(extra_feature),
                 bl_result[0],
                 bl_result[1], bl_result[2] + '/' + bl_result[3],
                 ' time:' + str(bl_result[4]) + str(' ') + bl_result[5]])
            print(table)

        if edge_feature_flag:
            X = normalize_(X_origin, axis=ax)
            print('With Distance feature')
            btwn_feature = normalize_(btwn_feature_origin, axis=ax)
            distance_feature = normalize_(distance_feature_orgin, axis=ax)
            # X = np.concatenate((X, distance_feature, btwn_feature), axis=1)
            X = np.concatenate((X, distance_feature), axis=1)
            X = normalize_(X, axis=ax);
            bl_result = baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag,
                                 laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)
            table.add_row(
                ['Node BL(deg)+Edge Feature ' + str(ax) + str(' ') + str(norm_flag_) + str(extra_feature), bl_result[0],
                 bl_result[1], bl_result[2] + '/' + bl_result[3],
                 ' time:' + str(bl_result[4]) + str(' ') + bl_result[5]])
            print(table)

    return table


# handle basepoint
def basepoint_filtration():
    np.random.seed(42)
    graphs = 10 * [1];
    dgms = 30 * [d.Diagram([(0, 1e-10)])];
    sub_dgms = 30 * [d.Diagram([(0, 1e-10)])];
    super_dgms = 30 * [d.Diagram([(0, 1e-10)])];

    # Initialization different base points
    n_basepoints = 1;
    s = 0;
    # sampleb = np.random.randint(1, 30, size=n_basepoints)
    sampleb = ['c'] + ['r'] * (n_basepoints - 1);
    sampleb = ['a']
    final_dgms = [d.Diagram()] * len(graphs_)

    # compute dgms
    for b in sampleb:
        seed = np.random.randint(1, 1000)
        (graphs[s], dgms[s], sub_dgms[s], super_dgms[s]) = \
            dgms_data(graph, np.array(beta), n_jobs, debug_flag, hop_flag='y', basep=b,
                      rs=seed)  # compute dgms for beta
        print(('Dgm in different iteration:'), end=' ')
        print_dgm(dgms[s][100])
        s += 1

    # aggregation different base points
    for b in range(n_basepoints):
        for i in range(len(graphs_)):
            final_dgms[i] = add_dgms(final_dgms[i], dgms[b][i])
    # export
    for i in range(len(final_dgms)):
        export_dgm(graph, final_dgms[i], i, filename='all_multi_basepoints', flag='ss')
    print('Finish exporting')

    dgms = dgms_data_(graph, n);
    dgms_summary(dgms);
    nulldgm = [d.Diagram()] * len(graphs_)
    pd_vector_data = clf_pdvector(0, (dgms, nulldgm, dgms), beta, Y, pd_flag, print_flag='off', nonlinear_flag=True,
                                  dynamic_range_flag=dynamic_range_flag)


def coarse_graph_i(i, key='deg'):
    print(('#'), end=' ')
    assert 'graphs_' in list(globals().keys())
    gs = graphs_[i]
    if key == 'deg':
        gg = [coarse_graph(g, print_flag=True) for g in gs]
        gg = [g for g in gg if g is not None]
        subgraphs = []
        for gi in gg:
            subgraphs += [gi.subgraph(c) for c in sorted(nx.connected_components(gi), key=len, reverse=True) if
                          len(c) > 3]
        # for g in subgraphs:
        #     print str(type(g))
        return subgraphs

    elif key == 'ricciCurvature':
        return [coarse_graph(g, key) for g in gs]


def coarse_graphs_():
    from joblib import delayed, Parallel
    coarsed_graphs_ = Parallel(n_jobs=-1)(delayed(coarse_graph_i)(i) for i in range(n))
    return coarsed_graphs_


def coarsed_pipeline0(i):
    import networkx as nx
    global coarsed_graphs
    result = []
    for gi in coarsed_graphs[i]:
        subgraphs = [gi.subgraph(c) for c in sorted(nx.connected_components(gi), key=len, reverse=True)]
        result += [function_basis(g, ['deg'], norm_flag='yes') for g in subgraphs if len(g) > 3]
    return result


def compute_coarsed_graphs_():
    from joblib import Parallel, delayed
    coarse_graphs_ = Parallel(n_jobs=-1)(delayed(coarsed_pipeline0)(i) for i in range(n))
    for attr in ['deg', '1_0_deg_max', '1_0_deg_min', '1_0_deg_mean', '1_0_deg_std']:
        assert list(nx.get_node_attributes(coarse_graphs_[0][0], attr).keys()) >= 0
    return coarse_graphs_


def set_node_filtration_param(beta, allowed):
    print(('Beta is %s' % (beta)), end=' ')
    if (str(type(beta)) == "<type 'numpy.ndarray'>") and (1 in beta):  # exclude hop related filtration
        beta_name = beta_dict[list(beta).index(1)]
        if beta_name not in allowed:
            pass
            # raise beta_name_not_in_allowed
        print(('Filtration is %s' % beta_name))
        hop_flag = 'n'
        basep = 0
    else:
        hop_flag = 'y'
        assert type(beta) == str
        assert beta[0:3] == 'hop'
        beta_name = beta
        basep = beta[4:]
    return (beta_name, hop_flag, basep)


def set_global_variable(graph):
    global allowed, allowed_edge, beta_dict, axis
    # allowed = ['deg', 'ricciCurvature', 'cc', 'fiedler']
    if graph == 'imdb_binary' or graph == 'imdb_multi':
        allowed = ['deg']
        # allowed = ['deg', 'ricciCurvature', 'label', 'cc']
    else:
        allowed = ['deg']
        # allowed = ['deg', 'ricciCurvature', 'label', 'cc']
    # allowed_edge = ['edge_ricci_minmax', 'edge_ricci_ave', 'edge_jaccard_minmax', 'edge_jaccard_ave', 'edge_p_minmax',
    #                 'edge_p_ave']
    allowed_edge = ['edge_jaccard_minmax', 'edge_jaccard_ave']
    beta_dict = {0: 'deg', 1: 'ricciCurvature', 2: 'fiedler', 3: 'cc', 4: 'label'};
    axis = 1  # 0 normalize feature and 1 normalize each sample


def check_global_safety():
    # assert (allowed ==  ['deg', 'ricciCurvature', 'cc', 'label', 'hop'])
    assert axis == 1


def coarse_graph(g, print_flag=True):
    # for some reason g might be none
    if g is None:
        return

    if print_flag:
        print('Before sampling:')
        print((nx.info(g)))

    nodelist = list(g.nodes)
    edgelist = list(g.edges)
    import random
    random.seed(42)
    n = len(nodelist) / 2
    deg = dict(nx.degree(g));
    n_edge = len(g.edges)
    for key, val in list(deg.items()):
        deg[key] = val / float((2 * n_edge))
    nodelist_ = random.sample(nodelist, int(n))
    # nodelist_ = numpy.random.choice(nodelist, int(n), deg.values())

    for i in range(n):
        u = nodelist_[i]
        if random.random() > 0.5:
            g.remove_node(u)
            continue
        vlist = random.sample(list(g.neighbors(u)), int(g.degree(u) / 2.0))
        for v in vlist:
            g.remove_edge(u, v)
    if print_flag:
        print('After sampling:')
        print((nx.info(g)))

    return g


def evaluate_tda_kernel(tda_kernel, Y, best_result_so_far):
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn import svm
    import time
    t1 = time.time()
    n = np.shape(tda_kernel)[0]
    grid_search_re = clf_search_offprint(np.zeros((n, 23)), Y, print_flag='off', kernel=tda_kernel,
                                         kernel_flag=True, nonlinear_flag=False)  # X is dummy here
    if grid_search_re['score'] < best_result_so_far[0] - 4:
        print('Saved one unnecessary evaluation of bad kernel')
        return (0, 0, {}, 0)

    cv_score = []
    for seed in range(5):
        clf = svm.SVC(kernel='precomputed', C=grid_search_re['param']['C'])
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, tda_kernel, Y, cv=k_fold, scoring='accuracy', n_jobs=-1)
        cv_score.append(scores.mean())

    cv_score = np.array(cv_score)
    t2 = time.time()
    svm_time = precision_format(t2 - t1, 1)
    return (
    precision_format(100 * cv_score.mean(), 1), precision_format(100 * cv_score.std(), 1), grid_search_re, svm_time)


def mem(graph, baseline, rfclf, clf_search_offprint, evaluate_best_estimator, function_basis, get_diagram, sw_parallel,
        evaluate_tda_kernel, clf_pdvector, merge_features):
    from joblib import Memory
    location = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/cachedir/'
    memory = Memory(location, verbose=1)

    baseline = memory.cache(baseline)
    rfclf = memory.cache(rfclf)
    clf_search_offprint = memory.cache(clf_search_offprint)
    evaluate_best_estimator = memory.cache(evaluate_tda_kernel)
    function_basis = memory.cache(function_basis)
    get_diagram = memory.cache(get_diagram)
    sw_parallel = memory.cache(sw_parallel)
    evaluate_tda_kernel = memory.cache(evaluate_tda_kernel)
    clf_pdvector = memory.cache(clf_pdvector)
    merge_features = memory.cache(merge_features)
    return (baseline, rfclf, clf_search_offprint, evaluate_best_estimator, function_basis, get_diagram, sw_parallel,
            evaluate_tda_kernel, clf_pdvector, merge_features)


def high_order_feature(n, high_order_graphs, order=2, axis=0):
    import numpy as np
    graphs_tmp = [high_order_graphs[i][order - 2] for i in range(len(graphs_))]
    n_bins = 40
    features = np.zeros((n, n_bins * 5))

    for i in range(n):
        g = graphs_tmp[i]
        # g = nx.gnp_random_graph(50, 0.5)
        for v in g.nodes():
            g.node[v]['deg'] = g.degree(v)
        for v in g.nodes():
            attribute_mean(g, v, 'deg')
        feature = np.zeros((1, 0))
        for attribute in ['1_0_deg_max', '1_0_deg_min', '1_0_deg_mean', '1_0_deg_std', 'deg']:
            lis = list(nx.get_node_attributes(g, attribute).values())
            feature_ = hisgram(lis, n_bin=n_bins, his_norm_flag='no', lowerbound=0, upperbound=50)
            feature_ = feature_.reshape(1, n_bins)
            feature = np.concatenate((feature, feature_), axis=1)
        features[i] = feature
    features = remove_zero_col(features)
    print(('The shape of deg is', np.shape(features)))
    return normalize_(features, axis=axis)


def remove_diagonal_point(dgm):
    diag = dgm2diag(dgm)
    diag = [point for point in diag if point[0] < point[1]]
    return diag2dgm(diag)


def fake_diagram(g, cardinality=2, attribute='deg', seed=42, true_dgm='null'):
    import networkx as nx
    from numpy import random
    import dionysus as d
    random.seed(seed)
    sample_pool = list(nx.get_node_attributes(g, attribute).values())
    if true_dgm != 'null':
        array_tmp = dgm2diag(true_dgm)
        sample_pool = [p[0] for p in array_tmp] + [p[1] for p in array_tmp]
    # assert 2*cardinality <= len(sample_pool) # relaxation
    try:
        sample = random.choice(sample_pool, size=2 * cardinality, replace=False)
    except:
        sample = random.choice(sample_pool, size=2 * cardinality, replace=True)
    assert set(sample).issubset(set(sample_pool))
    dgm = []
    for i in range(0, len(sample), 2):
        x_ = sample[i]
        y_ = sample[i + 1]
        dgm.append((min(x_, y_), max(x_, y_) + 1e-3))
    # print('fake dgm is', dgm)
    # print('ture dgm is');
    # use to test if fake dgm and real dgm are close in bd distance
    # print_dgm(true_dgm),
    # print ('bd distance of fake and true', d.bottleneck_distance(d.Diagram(dgm), remove_diagonal_point(true_dgm)))
    return d.Diagram(dgm)


def fake_diagrams(graphs_, dgms, true_dgms=['null'] * 10000, attribute='deg', seed=45):
    fake_dgms = []
    for i in range(len(graphs_)):
        cardinality = len(dgms[i])
        if len(graphs_[i]) == 0:
            fake_dgms.append(d.Diagram([(0, 0)]))
            continue
        tmp_dgm = fake_diagram(graphs_[i][0], cardinality=cardinality, attribute=attribute, seed=seed,
                               true_dgm=true_dgms[i])
        fake_dgms.append(tmp_dgm)
    return fake_dgms


def count_labels():
    labels = set()
    for i in range(10000):
        tmp_labels = set(nx.get_node_attributes(graphs_[i][0], 'label').values())
        labels = labels.union(tmp_labels)
        print((i, len(labels)))


def label_permutation(graphs_, permutation_dict):
    for i in range(len(graphs_)):
        for j in range(len(graphs_[i])):
            for n in graphs_[i][j].nodes():
                graphs_[i][j].node[n]['label_'] = permutation_dict[str(graphs_[i][j].node[n]['label'])]

    for i in range(len(graphs_)):
        for j in range(len(graphs_[i])):
            for n in graphs_[i][j].nodes():
                graphs_[i][j].node[n]['label'] = graphs_[i][j].node[n]['label_']

    return graphs_


def export_dgms(dgms):
    for i in range(len(dgms)):
        export_dgm(graph, dgms[i], i, filename=beta_name, flag='ss')
        export_dgm(graph, sub_dgms[i], i, filename=beta_name, flag='sub')
        export_dgm(graph, super_dgms[i], i, filename=beta_name, flag='super')
        export_dgm(graph, epd_dgms[i], i, filename=beta_name, flag='extended_one_homology')
    print(('Export dgm takes %s' % beta_name))


if __name__ == '__main__':
    (graph, rf_flag, _, high_order_flag, parameter_flag, pa, pb) = readargs()
    # graph = 'reddit_12K'; rf_flag='no'; pa = 0; pb = 1
    print(('pa and pb', pa, pb))
    print(('graph is %s' % graph))

    hyperparameter = set_hyperparameter()
    set_global_variable(graph)
    (hyperparameter_flag, norm_flag, loss_type, n_runs, pd_flag, multi_cv_flag, n_jobs, debug_flag, graph_isomorphisim,
     edge_fil, dynamic_range_flag) = hyperparameter
    threshold = threshold_data(graph)
    (table, n_row) = set_table(graph, hyperparameter);
    (data, Y_origin) = load_graph(graph);
    Y = Y_origin
    n = len(Y_origin)

    graphs_backup = compute_graphs_(['deg', 'ricciCurvature', 'cc'], graph_isomorphisim, graph, norm_flag='yes',
                                    feature_addition_flag=False)
    # graphs_backup = compute_graphs_(['edge_jaccard_minmax'], graph_isomorphisim, graph, norm_flag='yes', feature_addition_flag=False)
    graphs_ = graphs_backup
    print((graphs_backup[0][0].node[0]))
    beta = unit_vector(5, 3)
    if False:
        (sub_dgms, super_dgms, dgms, _) = handle_edge_filtration('edge_jaccard_minmax')
        beta_name = 'edge_jaccard_minmax'
        for i in range(len(graphs_)):
            export_dgm(graph, sub_dgms[i], i, filename=beta_name, flag='sub')
            export_dgm(graph, super_dgms[i], i, filename=beta_name, flag='super')
            export_dgm(graph, dgms[i], i, filename=beta_name, flag='ss')
            # print('Finish exporting ...', beta_name)
        print(sub_dgms)
        sys.exit()

    for false_label_percent in [0.0]:
        # for false_label_percent in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(('false_label_percent is %s\n\n' % false_label_percent))
        table.add_row([' '] * n_row);
        print(table)
        Y = change_label(graph, Y_origin, change_flag='no', false_label_percent=false_label_percent)
        print(Y)
        # (data, Y) = load_graph(graph);
        # Y_0 = change_label(graph, Y, change_flag='no', false_label_percent=0.0)
        # (data, Y) = load_graph(graph);
        # Y_1 = change_label(graph, Y, change_flag='no', false_label_percent=1.0)
        # (Y_0 == Y_1).all()
        ### edge filtration
        for edge_fil in allowed_edge:
            continue
            table = edge_baseline(edge_fil, graph, table)
        # baseline and stoa. Compute graphs_ once and use it for multiple times.
        if graph == 'imdb_binary' or graph == 'imdb_multi' or graph == 'dd_test' or graph == 'protein_data':
            norm_flag = 'no'
        else:
            norm_flag = 'yes'
        graphs_ = graphs_backup
        # graphs_ = compute_graphs_(allowed, graph_isomorphisim, graph, norm_flag='yes', feature_addition_flag=False);
        table = print_stoa(table, graph, n_row)
        for bin_size in [50, 70]:
            continue
            # table = node_baseline(graph, Y, graphs_, table, allowed, bin_size = bin_size, cdf_flag=True, edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False, edge_feature_flag=False, pro_flag=False, high_order_flag='False', laplacian_flag=False, lap_bandwidth=10)
            # for lap_bandwidth in [0.1, 1, 5, 10, 100]:
            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=True,
                                  edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                                  edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True,
                                  sanity_flag=True)

            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=True,
                                  edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                                  edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True,
                                  sanity_flag=False)

            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=False,
                                  edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                                  edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True,
                                  sanity_flag=True)

            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=False,
                                  edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                                  edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True,
                                  sanity_flag=False)

            # table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=False,
            #                       edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
            #                       edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True, sanity_flag=True)
        # table = node_baseline(graph, Y, graphs_, table, allowed, bin_size = bin_size, cdf_flag=False, edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False, edge_feature_flag=False, pro_flag=False, high_order_flag='False', laplacian_flag=False, lap_bandwidth=10)

    # node filtratillon
    norm_flag = 'yes'
    graphs_ = graphs_backup
    landscape_data = [];
    itr = 0;
    svm_data_hist = {0: {}}
    betalist = set_betalist(allowed)
    check_global_safety()
    for beta in betalist:
        assert len(betalist) == 3
        print(beta)
        try:
            (beta_name, hop_flag, basep) = set_node_filtration_param(beta, allowed)
        except beta_name_not_in_allowed:
            continue
        (graphs, dgms, sub_dgms, super_dgms, epd_dgms) = dgms_data(graph, beta, n_jobs, debug_flag, hop_flag=hop_flag,
                                                                   basep=basep, edge_fil='off')

        for i in range(len(graphs)):
            continue
            export_dgm(graph, sub_dgms[i], i, filename=beta_name, flag='sub')
            export_dgm(graph, super_dgms[i], i, filename=beta_name, flag='super')
            export_dgm(graph, dgms[i], i, filename=beta_name, flag='ss')
        print('Finish exporting...')

        if True:
            best_vec_result = 0
            for ax in [0, 1]:
                continue
                for epd_flag in [False]:
                    pd_vector_data = clf_pdvector(best_vec_result, (sub_dgms, super_dgms, dgms, epd_dgms), beta, Y,
                                                  epd_flag=epd_flag, pd_flag='True', print_flag='off',
                                                  nonlinear_flag=True, axis=ax, rf_flag=rf_flag,
                                                  dynamic_range_flag=dynamic_range_flag)  # use pd vector as baseline
                    print(('pd_vector_data is ', pd_vector_data))
                    table = add_row(table, pd_vector_data, beta_name, ax, filtration_type='node_vec')
                    print(table)

                    continue
                    for pvector in ['pi', 'pl']:
                        # pd_vector_data = clf_pdvector(best_vec_result, (sub_dgms, super_dgms, dgms, epd_dgms), beta, Y, epd_flag=epd_flag, pvec_flag=True, vec_type=pvector, pd_flag='True', multi_cv_flag=False, print_flag='off', nonlinear_flag=True, axis=ax, rf_flag=rf_flag, dynamic_range_flag=True)
                        # table = add_row(table, pd_vector_data, pvector, ax, filtration_type='node_vec')
                        pd_vector_data = clf_pdvector(best_vec_result, (fake_dgms, fake_dgms, fake_dgms, fake_dgms),
                                                      beta, Y,
                                                      epd_flag=epd_flag, pvec_flag=True, vec_type=pvector,
                                                      pd_flag='True', multi_cv_flag=False, print_flag='off',
                                                      nonlinear_flag=True, axis=ax, rf_flag=rf_flag,
                                                      dynamic_range_flag=True)
                        table = add_row(table, pd_vector_data, pvector, ax, filtration_type='node_vec')

                        print(table)

        for kernel_type in ['sw']:
            best_result_so_far = (0, 0, {})
            for bandwidth in kernel_parameter(kernel_type)['bw']:
                for k in kernel_parameter(kernel_type)['K']:
                    for p in kernel_parameter(kernel_type)['p']:
                        (true_kernel, _) = (tda_kernel, t1) = sw_parallel(dgms2swdgm(dgms), dgms2swdgm(dgms),
                                                                          parallel_flag=True,
                                                                          kernel_type=kernel_type, n_directions=10,
                                                                          bandwidth=bandwidth, K=k, p=p)
                        tda_kernel_data_ = evaluate_tda_kernel(tda_kernel, Y, best_result_so_far)
                        if tda_kernel_data_[0] > best_result_so_far[0]:
                            best_result_so_far = tda_kernel_data_  # only print best one
                            table.add_row([beta_name + ' ' + str(kernel_type) + ' bw:' + str(bandwidth), '', '',
                                           str(best_result_so_far[0]) + '/' + str(best_result_so_far[1]),
                                           'kct: ' + str(t1) + '/svm_time: ' + str(best_result_so_far[3]) + str(
                                               best_result_so_far[2])])
                            print(table)

        if beta[1] == 1:
            X = pairing_feature(dgms, n_bin=30, range_=(-1, 1))
        else:
            X = pairing_feature(dgms, n_bin=20, range_=(0, 1))
        param = searchclf(X, Y, 1002, test_size=0.1)
        evaluate_clf(X, Y, param, n_splits=10)

        X = aggregation_feature(dgms)
        print(X)
        param = searchclf(X, Y, 1002, test_size=0.1)
        evaluate_clf(X, Y, param, n_splits=10)
        continue

        table = add_row(table, 'dummy', 'dummy', 'dummy', filtration_type='empty')
        print(('check graphs_'), end=' ')
        print((graphs_[0][0].node[0]))
        test_norm(super_dgms)
        (matrix_prl, flag3) = load_data(graph, 'matrix_prl', beta)  # load existing matrix
        if flag3 != 'success':
            matrix_prl = Parallel(n_jobs=n_jobs)(
                delayed(get_matrix_i)(i, debug=debug_flag) for i in range(len(dgms)))  # compute matrix in parallel
            dump_data(graph, matrix_prl, 'matrix_prl', beta)
        print('Finish matrix_prl')

        (dist_matrix, idx_dict) = format_matrixprl(matrix_prl)

        if False:
            diags = [dgm2diag_(i) for i in dgms]
            for sigma in [0.1, 1, 10, 100][1:2]:
                dist_matrix = get_roland_matrix_prl(diags, sigma)
                for c in [0.01, 0.1, 1, 10, 100]:
                    kernelsvm(dist_matrix, Y, c, sigma, dist_flag='no')
                    # (kernel, svm_data) = kernelsvm(dist_matrix, Y, c, sigma, dist_flag='no');
        # continue
        c = 0.1;
        sigma = 0.1
        (kernel, svm_data) = kernelsvm(dist_matrix, Y, c, sigma, dist_flag='yes')
        title_cache = (beta, graph, round, svm_data[0]['train_acc'], svm_data[0]['test_acc'], c, sigma)
        # MDS(dist_matrix, Y, title_cache, gd='True', print_flag='True')
        # continue
        svm_data_hist[itr] = svm_data
        gddata = get_gddata(svm_data)  # yyhat is a dict. yyhat['y'] is a 10-list of numpy array
        cache = (kernel, gddata['y'][0], gddata['y_hat'][0], beta,
                 gddata['alpha'][0])  # [0] here is the index of training number
        if itr > 1:
            check_cache = (svm_data_hist[itr - 1], svm_data_hist[itr]);
            check_trainloss(check_cache)

        title_cache = (beta, graph, round, svm_data[0]['train_acc'], svm_data[0]['test_acc'], c, sigma)
        # MDS(dist_matrix, Y, title_cache, gd='True', print_flag='True')
        train_idx = svm_data[0]['train_idx']
        gradient_ = total_gradient(cache, train_idx, sigma)
        gradient_[0] = 0;
        gradient_[2] = 0;
        gradient_[4] = 0
        total_loss(itr, cache, train_idx)
        landscape_data += [{'beta': beta, 'others': svm_data, 'pd_vector_data': pd_vector_data}]
        write_landscapedata(graph, {'beta': beta, 'others': svm_data})

        beta = beta_update(beta, gradient_)
        beta[0] = 0;
        beta[2] = 0;
        beta[4] = 0;
        beta = beta / float(sum(beta))
        continue

        print('----------------------------------------------------------\n')
        best8 = svm_hyperparameter(dist_matrix, Y, hyperparameter_flag='no')
        title_cache = (beta, graph, round, best8[0][2], best8[0][3], best8[0][0], best8[0][1])
        MDS(dist_matrix, Y, title_cache)

    sys.exit()
