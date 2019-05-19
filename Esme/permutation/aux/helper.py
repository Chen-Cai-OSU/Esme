from functools import wraps
from aux.tools import make_direct, dgm2diag, remove_zero_col, print_dgm, flip_dgm
import numpy as np
import sys
import networkx as nx
from prettytable import PrettyTable

sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/pythoncode')
sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence')
sys.path.append('/home/cai.507/Documents/Utilities/dionysus/build/bindings/python')
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode')
sys.path.append('/home/cai.507/Documents/DeepLearning/GraphSAGE')
def viz_pd(graphs_, i, beta=np.array([0,0,0,0,1])):
    import matplotlib.pyplot as plt
    k = 0
    g = graphs_[i][k]
    f, axarr = plt.subplots(2, sharex=True)

    assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
    g = fv(g, beta, hop_flag='n', basep=0)  # belong to pipe1
    (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='max')  # belong to pipe1
    dgm_sub = get_diagram(g, key='fv', subflag='True')
    print_dgm(dgm_sub)
    (dgm_title, x, y) = scatter_dgm(dgm_sub)

    axarr[0].set_title(dgm_title)
    axarr[0].scatter(x, y)
    nx.draw(g, pos=nx.spring_layout(g), node_size = 5)
    plt.draw()
    plt.title('Graph: ' + str(i) + str(beta))
    filename = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/interpolation/'
    make_direct(filename)
    filename = filename + str(beta) + '.png'
    plt.savefig(filename)
    # plt.show()
    plt.close()
class Nodeid_Unfound_Error(Exception):
    pass
def import_others():
    import dionysus as d
    import numpy as np
    from joblib import Parallel, delayed
    # from progress.bar import Bar
    # global bar
    # bar = Bar('Processing', max=100)
def convert2nx(graph, i, print_flag='False'):
    # graph: python dict
    import networkx as nx
    keys = list(graph.keys())
    try:
        assert keys == list(range(len(list(graph.keys()))))
    except AssertionError:
        # pass
        print(('%s graph has non consecutive keys'%i))
        print('Missing nodes are the follwing:')
        for i in range(max(graph.keys())):
            if i not in list(graph.keys()):
                print(i),
        # print('$') # modification for reddit binary
    gi = nx.Graph()
    # add nodes
    for i in keys:
        gi.add_node(i) # change from 1 to i. Something wired here

    assert len(gi) == len(keys)
    # add edges
    for i in keys:
        for j in graph[i]['neighbors']:
            if j > i:
                gi.add_edge(i, j)
    # add labels
    # print(gi.node)
    for i in keys:
        # print graph[i]['label']
        if graph[i]['label']=='':
            gi.node[i]['label'] = 1
            # continue
        try:
            gi.node[i]['label'] = graph[i]['label'][0]
        except TypeError: # modifications for reddit_binary
            gi.node[i]['label'] = graph[i]['label']
        except IndexError:
            gi.node[i]['label'] = 0 # modification for imdb_binary
    # print(gi)
    # print (gi.vs[1])
    assert len(gi.node) == len(list(graph.keys()))
    gi.remove_edges_from(gi.selfloop_edges())
    if print_flag=='True':
        print(('graph: %s, n_nodes: %s, n_edges: %s' %(i, len(gi), len(gi.edges)) ))

    return gi
def get_subgraphs(g, threshold=1):
    # nci1 60th graph is a disconnected graphs with three components(19,4,4)
    # test_data = {0: {'neighbors': [13], 'label': (25,)}, 1: {'neighbors': [15], 'label': (25,)}, 2: {'neighbors': [12, 20], 'label': (1,)}, 3: {'neighbors': [14, 21], 'label': (1,)}, 4: {'neighbors': [23], 'label': (1,)}, 5: {'neighbors': [24], 'label': (1,)}, 6: {'neighbors': [17], 'label': (1,)}, 7: {'neighbors': [19], 'label': (1,)}, 8: {'neighbors': [23], 'label': (1,)}, 9: {'neighbors': [24], 'label': (1,)}, 10: {'neighbors': [11, 17], 'label': (2,)}, 11: {'neighbors': [10, 12, 13], 'label': (3,)}, 12: {'neighbors': [ 2, 11, 15], 'label': (3,)}, 13: {'neighbors': [ 0, 11, 14], 'label': (3,)}, 14: {'neighbors': [ 3, 13, 16], 'label': (3,)}, 15: {'neighbors': [ 1, 12, 16], 'label': (3,)}, 16: {'neighbors': [14, 15], 'label': (3,)}, 17: {'neighbors': [ 6, 10, 18], 'label': (3,)}, 18: {'neighbors': [17, 19], 'label': (3,)}, 19: {'neighbors': [ 7, 18, 22], 'label': (3,)}, 20: {'neighbors': [2], 'label': (3,)}, 21: {'neighbors': [3], 'label': (3,)}, 22: {'neighbors': [19], 'label': (3,)}, 23: {'neighbors': [ 4,  8, 25], 'label': (3,)}, 24: {'neighbors': [ 5,  9, 26], 'label': (3,)}, 25: {'neighbors': [23], 'label': (3,)}, 26: {'neighbors': [24], 'label': (3,)}}
    # test_graphs = get_subgraphs(convert2nx(test_data))
    # for each graph(possibly disconnected), get all the subgraphs whose size large than 4
    # g = convert2nx(g)
    assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>"
    import networkx as nx
    # nx.is_connected(g)
    # get subgraph
    subgraphs = [g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    # subgraphs = [nx.convert_node_labels_to_integers(g) for g in subgraphs]
    subgraphs = [c for c in subgraphs if len(c) > threshold]
    return subgraphs

def timefunction(method, time_flag=False):
    def timed(*args, **kw):
        import time
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if (te-ts)>1:
                print(('%r  %2.2f s' % (method.__name__, (te - ts) )))
        if time_flag == False:
            return result
        else:
            return result, te-ts
    return timed

def timefunction_precise(method):
    def timed(*args, **kw):
        import time
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if (te-ts)>0.1:
                print(('%r  %2.2f s' % (method.__name__, (te - ts) )))
        return result
    return timed

def find_nbrs(g, i):
    return g.nodes[i]

def count_check(function, count=[0]):
    """Returns number of times any function with this decorator is called
    """
    @wraps(function)
    def increase_count(*args, **kwargs):
        count[0] += 1
        return function(*args, **kwargs), count[0]

    return increase_count

def get_matrix_i(i, dtype='bottleneck', debug='off', method='grey'):
    global dgms
    if debug == 'on':
        print(('Processing %s row'%i))
    # from dgms(a list of pds), get the i-th row of bd distance matrix
    import numpy as np
    import dionysus as d
    n = len(dgms)
    dist_vec = np.zeros((1, n))
    idx_dict = {}
    if dtype == 'bottleneck':
        # for j in range(n):
        for j in range(i, n):
            if debug=='on':
                print((i, j))
            if method == 'grey':
                dist = d.bottleneck_distance(dgms[i], dgms[j], delta = 0.001, compute_longest_edge=True)
                # print(i, j, dist)
                # sys.exit()
            elif method == 'me':
                bd = d.bottleneck_distance(dgms[i], dgms[j], delta = 0.001)
                dist = d_bottleneck_distance_with_edge(dgms[i], dgms[j], bd, debug=debug_flag)
            # print(dist)
            dist_vec[0, j] = dist[0]
            (idx1, idx2) = dist[1]

            # deal with four cases
            if ((str(idx1) == '-1') and (str(idx2) == '-1')):
                idx_dict[(i, j)] = 'same'

            elif ((str(idx1) != '-1') and (str(idx2) != '-1')):
                idx_dict[(i, j)] = (
                (dgms[i][idx1].birth, dgms[i][idx1].death), (dgms[j][idx2].birth, dgms[j][idx2].death))

            elif ((str(idx1) != '-1') and (str(idx2) == '-1')):
                idx_dict[(i, j)] = ((dgms[i][idx1].birth, dgms[i][idx1].death), None)

            elif ((str(idx1) == '-1') and (str(idx2) != '-1')):
                idx_dict[(i, j)] = (None, (dgms[j][idx2].birth, dgms[j][idx2].death))

            else:
                raise AssertionError
                # assert np.amax(abs((dist_matrix - dist_matrix.T))) < 0.02 # db is not symmetric
        # print('.'),
    return (dist_vec, idx_dict)
def get_matrix_i_prl_list(i, dist_matrix, dtype='bottleneck', debug='off', method='grey', array_flag='False'):
    # dgms = [diag2dgm(diag) for diag in diag_list]
    print(('Finish processing %s'%i))
    return get_matrix_i_prl(i, dist_matrix, dtype='bottleneck', debug='off', method='grey', array_flag='False')
def computePD(i):
    import dionysus as d
    import numpy as np
    # np.random.seed(42)
    f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
    m1 = d.homology_persistence(f1)
    dgms1 = d.init_diagrams(m1, f1)
    return dgms1[1]
def test_dionysus_prl():
    def get_dgms_(n_jobs=1):
        from joblib import delayed, Parallel
        return Parallel(n_jobs=n_jobs)(delayed(computePD)(i) for i in range(10))

    return get_dgms_(2)
def get_matrix_i_prl(i, dist_matrix, dtype='bottleneck', debug='off', method='grey', array_flag='False'):
    import os
    print(('Worker %s, Start processing %s' % (os.getpid(), i)))
    if debug == 'on':
        print(('Processing %s row'%i))
    # from dgms(a list of pds), get the i-th row of bd distance matrix
    import numpy as np
    assert 'dgms' in globals()
    import dionysus as d
    n = len(dgms)
    idx_dict = {}
    if dtype == 'bottleneck':
        # for j in range(n):
        for j in range(i, n):
            if debug == 'on':
                print((i, j))
            if method == 'grey':
                dist = d.bottleneck_distance(dgms[i], dgms[j], delta = 0.001)
                # print(i,j),
                # print('First half: bd computation is OK')
            dist_matrix[i, j] = dist
        for j in range(n-1-i, n):
            dist = d.bottleneck_distance(dgms[n-1-i], dgms[j], delta=0.001)
            dist_matrix[n-1-i, j] = dist
            # print(n-1-i, j),
            # print('Second half: bd computation is OK')
        print(('.'), end=' ')
    print(('Worker %s, Finishes processing %s' % (os.getpid(), i)))
    return (dist_matrix, idx_dict)
def test_matrix_prl():
    import tempfile
    import shutil
    import os
    import numpy as np
    from joblib import Parallel, delayed
    from joblib import load, dump
    assert 'dgms' in globals()
    n = len(dgms)
    folder = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/joblib'
    # folder = '/Users/admin/Documents/osu/Research/deep-persistence'
    # dgmsave = os.path.join(folder, 'dgms')
    dist_matrix_name = os.path.join(folder, 'dist_matrix')

    # Pre-allocate a writeable shared memory map as a container for the
    # results of the parallel computation
    dist_matrix = np.memmap(dist_matrix_name, dtype=float,
                     shape=(n,n), mode='w+')

    # Dump the input data to disk to free the memory
    dgms_list = [dgm2diag(dgm) for dgm in dgms]
    # dump(dgms_list, dgmsave)
    # dgms = load(dgmsave, mmap_mode='r')

    # Fork the worker processes to perform computation concurrently
    Parallel(n_jobs=-1)(delayed(get_matrix_i_prl)(i, dist_matrix)
                       for i in range(int(n/2.0)-1))

    print(dist_matrix)
    return dist_matrix
def test_memmap():
    from joblib import load, dump
    import dionysus as d
    import tempfile
    import os
    import numpy as np
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib_test.mmap')
    if os.path.exists(filename): os.unlink(filename)
    # _ = dump(d.Diagram([(1,2),(3,4)]), filename)
    _ = dump(np.array(([[1,2],[3,4]])), filename)
    large_memmap = load(filename, mmap_mode='r+')
    large_memmap +=1
    return large_memmap
def get_diagram(g, key='fv', typ='tuple', subflag = 'True', one_homology_flag=False):
    # only return 0-homology of sublevel filtration
    # type can be tuple or pd. tuple can be parallized, pd cannot.
    import dionysus as d
    def get_simplices(gi, key='fv'):
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        import networkx as nx
        assert len(list(gi.node)) > 0
        assert key in list(gi.node[list(gi.nodes)[0]].keys())

        simplices = list()
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2][key]):
            tup = ([u, v], data[key])
            simplices.append(tup)

        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            tup = ([v], data[key])
            simplices.append(tup)

        return simplices

    simplices = get_simplices(g)
    @timefunction
    def compute_EPD(g__, pd_flag=False, debug_flag=False):
        w = -1
        import dionysus as d
        values = nx.get_node_attributes(g__, 'fv')
        simplices = [[x[0], x[1]] for x in list(g__.edges)] + [[x] for x in g__.nodes()]
        up_simplices = [d.Simplex(s, max(values[v] for v in s)) for s in simplices]
        down_simplices = [d.Simplex(s + [w], min(values[v] for v in s)) for s in simplices]
        if pd_flag==True:
            down_simplices = [] # mask the extended persistence here

        up_simplices.sort(key=lambda s1: (s1.dimension(), s1.data))
        down_simplices.sort(reverse=True, key=lambda s: (s.dimension(), s.data))
        f = d.Filtration([d.Simplex([w], -float('inf'))] + up_simplices + down_simplices)
        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)
        if debug_flag==True:
            print('Calling compute_EPD here with success')
            print([print_dgm(dgm) for dgm in dgms])
        return dgms
    if one_homology_flag==True:
        epd_dgm = compute_EPD(g, pd_flag=False)[1]
        def post_process(dgm, debug_flag=False):
            import dionysus as d
            if len(dgm)==0:
                return d.Diagram([(0,0)])
            # print_dgm(dgm)
            # print(type(dgm))
            # print(len(dgm))
            for p in dgm:
                if p.birth==np.float('-inf'):
                    p.birth = 0
                if p.death == np.float('inf'):
                    p.death = 0
            if debug_flag==True:
                print(('Before flip:'), end=' ')
                print_dgm(dgm)
            dgm = flip_dgm(dgm)
            if debug_flag==True:
                print(('After:'), end=' ')
                print_dgm(dgm)
            return dgm
        epd_dgm = post_process(epd_dgm)
        return epd_dgm

    def compute_PD(simplices, sub=True, inf_flag='False'):
        import dionysus as d
        def compare(s1, s2, sub_flag=True):
            if sub_flag == True:
                if s1.dimension() > s2.dimension():
                    return 1
                elif s1.dimension() < s2.dimension():
                    return -1
                else:
                    return cmp(s1.data, s2.data)
            elif sub_flag == False:
                return -compare(s1, s2, sub_flag=True)

        f = d.Filtration()
        for simplex, time in simplices:
            f.append(d.Simplex(simplex, time))
        if sub == True:
            f.sort()
        elif sub == False:
            f.sort(reverse=True)
        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)

        def del_inf(dgms):
            import dionysus as d
            dgms_list = [[], []]
            for i in range(2):
                pt_list = list()
                for pt in dgms[i]:
                    if (pt.birth == float('inf')) or (pt.death == float('inf')):
                        pass
                    else:
                        pt_list.append(tuple([pt.birth, pt.death]))
                # pt_list.append(tuple([0, 0])) # append a dummy point
                diagram = d.Diagram(pt_list)
                dgms_list[i] = diagram
            return dgms_list

        if inf_flag == 'False':
            dgms = del_inf(dgms)

        # for some degenerate case, return dgm(0,0)
        if (dgms == []) or (dgms == None):
            return d.Diagram([[0,0]])
        # print_dgm(dgms)
        return dgms

    super_dgms = compute_PD(simplices, sub=False)
    # print_dgm(super_dgms[0])
    # print_dgm(super_dgms[1])

    sub_dgms = compute_PD(simplices, sub=True)
    n_node = len(g.nodes())
    _min = min([g.nodes[n][key] for n in g.nodes])
    _max = max([g.nodes[n][key] for n in g.nodes])+ 1e-5 # avoid the extra node lies on diagonal
    p_min = d.Diagram([(_min, _max)])
    p_max = d.Diagram([(_max, _min)])

    # print(p[0])
    sub_dgms[0].append(p_min[0])
    super_dgms[0].append(p_max[0])
    # add super_level filtration
    # for p in super_dgms[0]:
    #     sub_dgms[0].append(p)
    if subflag=='True':
        return sub_dgms[0]
    elif subflag=='False':
        return super_dgms[0]
    # no longer needed since mrzv has fix the error
    # tuple_dgms = [(p.birth, p.death) for p in sub_dgms[0]]

    # return sub_dgms[0]
    # if typ == 'tuple':
    #     return tuple_dgms

def single_test(kernel, c, Y, r_seed, lbd = 1e-7, loss='hinge', debug='off', debug_flag='off', test_flag=False):
    import  numpy as np
    def compute_alpha_kernel(kernel, Y, lbd):
        import numpy as np
        n = np.shape(Y)[0]
        inv_K = np.linalg.inv(kernel + lbd * np.eye(n))
        alpha = np.dot(inv_K, Y)
        assert len(alpha) == n
        return alpha
    def get_alpha(clf, n):
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
            alpha[i] = dual_coef[0][k]  # need to change if want to handle multiple classes
            k = k + 1
        if debug == 'on':
            print(dual_coef)
            print((clf.n_support_))
            print((k, np.sum(clf.n_support_)))
        # assert k ==np.sum(clf.n_support_)
        assert k == len(clf.support_)
        return alpha

    from sklearn import svm
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score
    n = np.shape(Y)[0]
    test_accuracies = []
    train_accuracies = []

    print(('.'), end=' ')
    if test_flag == True:
        y_train, y_test, indices_train, indices_test = train_test_split(Y, list(range(n)), test_size=1, random_state=r_seed)
    elif test_flag == False:
        y_train, y_test, indices_train, indices_test = train_test_split(Y, list(range(n)), test_size=0.1, random_state=r_seed)
    n_train = len(y_train)
    # kernel = np.exp(np.multiply(dist_matrix, -dist_matrix) / sigma)
    kernel_train = kernel[np.ix_(indices_train, indices_train)]
    assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
    kernel_test = kernel[np.ix_(indices_test, indices_train)]
    kernel_all = kernel[np.ix_(list(range(n)), indices_train)]
    np.shape(kernel_test)

    if loss == 'hinge':
        clf = svm.SVC(kernel='precomputed', C=c)
        clf.fit(kernel_train, y_train)
        y_pred = clf.predict(kernel_test)
        y_reg = y_pred # need to change later
        alpha = get_alpha(clf, n_train)
        y_pred_train = clf.predict(kernel_train)
        # print('Train accuracy is %s'%accuracy_score(y_train, y_pred_train))

    elif loss == 'square':
        lbd = lbd
        alpha = compute_alpha_kernel(kernel_train, y_train, lbd)
        K = kernel_test
        assert np.shape(K)[1] == len(alpha)
        y_pred = np.dot(kernel_test, alpha)
        y_pred_train = np.sign(np.dot(kernel_train, alpha))
        if debug=='on':
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
        alpha = get_alpha(clf, n_train)

        y_pred_train = np.sign(clf.predict(kernel_train))
        # print('Train accuracy is %s' % accuracy_score(y_train, y_pred_train))

    train_acc = (accuracy_score(y_train, y_pred_train))
    if debug_flag == 'on':
        for  y_pred_train_i, y_train_i, i in zip(y_pred_train, y_train, indices_train):
          # if y_pred_train_i!= y_train_i:
          #   print(i, 'has been classified as ', y_pred_train_i, 'and should be ', y_train_i)
            if (y_pred_train_i!= y_train_i and y_pred_train_i == 1):
                print((i), end=' ')
                print((','), end=' ')
    test_acc = (accuracy_score(y_test, y_pred))

    test_accuracies.append(accuracy_score(y_test, y_pred))
    train_accuracies.append(accuracy_score(y_train, y_pred_train))
    assert 0 not in test_accuracies
    assert 0 not in train_accuracies
    return {'reg': y_reg, 'test_idx': indices_test, 'train_idx': indices_train, 'coef': alpha, 'train_acc': train_acc, 'test_acc': test_acc,
            'y_hat':clf.decision_function(kernel_all), 'y': Y, 'kernel': kernel, 'kernel_train': kernel_train, 'kernel_test': kernel_test, 'clf':clf}
    # return {'train_acc': train_test, 'test_acc': test_acc}
def get_accuracies(data, c, sigma, threshold, loss='hinge'):
    # a list of dict
    import numpy as np
    train_accuracies = []
    test_accuracies = []
    n = len(data)
    for i in range(n):
        train_accuracies.append(data[i]['train_acc'])
        test_accuracies.append(data[i]['test_acc'])

    if 100*np.mean(test_accuracies) < threshold:
        return

    if (loss == 'hinge') or (loss == 'svr'):
        # print('{}: c is {}, after {} runs, mean test accuracy is {:1.3f}, std is {:1.3f}'.format(loss, c, n, 100*np.mean(test_accuracies), 100*np.std(test_accuracies)))
        # print('{}: c is {}, after {} runs, mean train accuracy is {:1.3f}, std is {:1.3f}'.format(loss, c, n, 100*np.mean(train_accuracies), 100*np.std(train_accuracies)))
        return {'c': c, 'sigma': sigma, 'n_runs':n, 'test_accuracies': test_accuracies, 'train_accuracies': train_accuracies,
                'train_acc': np.mean(train_accuracies)*100, 'train_std': np.std(train_accuracies)*100,
                'test_acc': np.mean(test_accuracies)*100, 'test_std': np.std(test_accuracies)*100}
    elif loss == 'square':
        print(('{}:  mean train accuracy is {:1.3f}, std is {:1.3f}'.format(loss, 100 * np.mean(train_accuracies),100 * np.std(train_accuracies))))  #
        print(('{}:  mean test accuracy is {:1.3f}, std is {:1.3f}' .format(loss, 100*np.mean(test_accuracies), 100*np.std(test_accuracies)))) #
def get_gddata(data):
    y = []
    y_hat = []
    alpha = []
    n = len(data)
    for i in range(n):
        y_tmp = [label for label in data[i]['y']]
        y.append(y_tmp)
        y_hat.append(data[i]['y_hat'])
        alpha.append(data[i]['coef'])
    return {'y': y, 'y_hat': y_hat, 'alpha': alpha}
def alpha_svm(dist_matrix, Y, alpha, train_idx, c, sigma):
    import numpy as np
    kernel = np.exp(np.multiply(dist_matrix, -dist_matrix) / sigma)
    y_hat = np.dot(kernel, alpha)
    return y_hat
def test_bdkernel(kernel, c, Y, lbd = 1e-7, loss='hinge', debug='off'):
    test_accuracies = list()
    train_accuracies = list()
    for i in range(10):
        data = single_test(kernel, c, Y, lbd = 1e-7, loss='square', debug='off')
        test_accuracies.append(data['test_acc'])
        train_accuracies.append(data['train_acc'])
def attribute_mean(g, i, key='deg', cutoff=1, iteration=0):
    # g = graphs_[i][0]
    # g = graphs_[0][0]
    # attribute_mean(g, 0, iteration=1)
    # g.nodes[0]
    # g = graphs_[100][0]
    # g_ = attribute_mean(g, 0)
    import networkx as nx
    for itr in [iteration]:

        assert key in list(g.node[i].keys())
        # nodes_b = nx.single_source_shortest_path_length(g,i,cutoff=cutoff).keys()
        # nodes_a = nx.single_source_shortest_path_length(g,i,cutoff=cutoff-1).keys()
        # nodes = [k for k in nodes_b if k not in nodes_a]
        nodes = list(g[i].keys())

        if iteration == 0:
            nbrs_deg = [g.node[j][key] for j in nodes]
        else:
            key_ = str(cutoff) + '_' + str(itr-1) + '_' + key +  '_' + 'mean'
            nbrs_deg = [g.node[j][key_] for j in nodes]
            g.node[i][ str(cutoff) + '_' + str(itr) + '_' + key] = np.mean(nbrs_deg)
            return

        oldkey = key
        key = str(cutoff) + '_' + str(itr) + '_' + oldkey
        key_mean = key + '_mean'; key_min = key + '_min'; key_max = key + '_max'; key_std = key + '_std'
        key_sum = key + '_sum'
        # key_3q = key + '_3q'; key_1q = key + '_1q'; key_
        # key_secondmax = key + '_secondmax'; key_secondmin = key + '_secondmin';
        # key_thirdmax = key + '_thirdmax'; key_thirdmin = key + '_thirdmin';

        if len(nbrs_deg) == 0:
            g.node[i][key_mean] = 0
            g.node[i][key_min] = 0
            g.node[i][key_max] = 0
            g.node[i][key_std] = 0
            g.node[i][key_sum] = 0
            # g.nodep[i][key_3q] = 0
            nbrs_deg_sorted = [0]
        else:
            # assert np.max(nbrs_deg) < 1.1
            g.node[i][key_mean] = np.mean(nbrs_deg)
            g.node[i][key_min] = np.min(nbrs_deg)
            g.node[i][key_max] = np.max(nbrs_deg)
            g.node[i][key_std] = np.std(nbrs_deg)
            g.node[i][key_sum] = np.sum(nbrs_deg)
            # g.node[i][key_1q] = np.percentile(nbrs_deg, 25)
            # g.node[i][key_3q] = np.percentile(nbrs_deg, 75)

        # try:
        #     g.node[i][key_secondmax] = nbrs_deg_sorted[1]
        # except:
        #     g.node[i][key_secondmax] = 0
        # try:
        #     g.node[i][key_secondmin] = nbrs_deg_sorted[-2]
        # except:
        #     g.node[i][key_secondmin] = 0
        #
        # try:
        #     g.node[i][key_thirdmax] = nbrs_deg_sorted[2]
        # except:
        #     g.node[i][key_thirdmax] = 0
        #
        # try:
        #     g.node[i][key_thirdmin] = nbrs_deg_sorted[-3]
        # except:
        #     g.node[i][key_thirdmin] = 0

def sample_data(data, Y):
    idx = list(range(0, 500)) + list(range(4000, 4500))
    d_slice = {key: value for key, value in list(data.items())
               if key in idx}
    for i in range(500, 1000):
        d_slice[i] = d_slice.pop(i + 3500)

    Y_slice = Y[idx]
    assert max(d_slice.keys()) == 999
    return (d_slice, Y_slice)
def functionongraph(graphs_, i, key='deg', edge_flag=False, short_cut_flag=False):
    # for graphs_[i], get the key-val distribution
    components = len(graphs_[i]); lis = []
    for j in range(components):
        g = graphs_[i][j]
        try:
            assert (str(type(g)) ==  "<class 'networkx.classes.graphviews.SubGraph'>") or (str(type(g))) == "<class 'networkx.classes.graph.Graph'>"
        except AssertionError:
            if g is None:
                print('wired case: g is None')
                return [0]
            else:
                print('Unconsidered Cases in function on graph')
        if edge_flag==False:
            tmp = [g.nodes[k][key] for k in g.nodes]
        elif edge_flag==True:
            if key[-7:]=='_minmax':
                minkey = key[:-6]+'min'
                maxkey = key[:-6] + 'max'
            elif key[-4:]=='_ave':
                minkey = key
                maxkey = key

            tmp_min = [g.node[k][minkey] for k in g.nodes if ('type' in list(g.nodes[k].keys()) and g.node[k]['type']=='edge_node')]
            tmp_max = [g.node[k][maxkey] for k in g.nodes if ('type' in list(g.nodes[k].keys()) and g.node[k]['type']=='edge_node')]
            tmp = tmp_min + tmp_max
        lis += tmp

    return lis

def hisgram_single_feature(graphs_, n_bin, key='deg', his_norm_flag='yes', edge_flag=False, lowerbound=-1, upperbound=1, cdf_flag=False, uniform_flag = True):
    import numpy as np
    n = len(graphs_)
    feature_vec = np.zeros((n, n_bin))
    for i in range(n):
        lis = functionongraph(graphs_, i, key, edge_flag=edge_flag)
        if lis == []:
            feature_vec[i] = 0
        feature_vec[i] = hisgram(lis, n_bin, his_norm_flag=his_norm_flag, lowerbound=lowerbound, upperbound=upperbound, cdf_flag=cdf_flag, uniform_flag=uniform_flag)
    return feature_vec

def hisgram(lis, n_bin=100, his_norm_flag='yes', lowerbound=-1, upperbound=1, cdf_flag=False, uniform_flag=True):
    import numpy as np
    if lis == []:
        print ('lis is empty')
        return [0]*n_bin
    # normalize lis
    # needs to be more rigirous
    if his_norm_flag == 'yes':
        try:
            assert max(lis) < 1.1 # * 100000 # delelte 100 later
        except AssertionError:
            print((max(lis)), end=' ')
    # hisgram(functionongraph(graphs_, 3556, 'fiedler'))
        assert min(lis) > -1.1

    if uniform_flag == False:
        assert lowerbound + 1e-3 > 0
        n_bin_ = np.logspace(np.log(lowerbound + 1e-3), np.log(upperbound),n_bin+1, base = np.e)
    elif uniform_flag == True:
        n_bin_ = n_bin

    if cdf_flag == True:
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(lis)
        if uniform_flag == True:
            return ecdf([i / np.float(n_bin) for i in range(0, n_bin)])
        elif uniform_flag == False:
            return ecdf([i / np.float(n_bin) for i in range(0, n_bin)])

    result = np.histogram(lis, bins=n_bin_, range=(lowerbound,upperbound))
    return result[0]

# import numpy as np
# np.logspace(np.log(1e-4), np.log(1),30, base = np.e)
@timefunction_precise
def merge_features(graph, graphs_, allowed, n_bin=30, his_norm_flag='yes', edge_flag=False, cdf_flag=False, uniform_flag = True):
    print(('Number of bins are %s'%n_bin))
    n = len(graphs_)
    X = np.zeros((n, 1))
    for key in allowed:
        print(key)
        if (key=='label') :
            if graph == 'dd_test':
                nbin = 90
            else:
                nbin = 40
            tmp = hisgram_single_feature(graphs_, nbin, 'label', his_norm_flag=his_norm_flag, edge_flag=edge_flag, lowerbound=0, upperbound=1, cdf_flag=cdf_flag, uniform_flag=uniform_flag)

        elif key == 'ricciCurvature': # use default bound for ricci curvature
            tmp = hisgram_single_feature(graphs_, n_bin, key, his_norm_flag=his_norm_flag, edge_flag=edge_flag, cdf_flag=cdf_flag, uniform_flag=uniform_flag)
        else:
            tmp = hisgram_single_feature(graphs_, n_bin, key, his_norm_flag=his_norm_flag, edge_flag=edge_flag, cdf_flag=cdf_flag, uniform_flag=uniform_flag, lowerbound=0)
        X = np.append(X, tmp, axis=1)
    return remove_zero_col(X[:,1:])

def set_baseline_directory(graph):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + str(graph) + '/Baseline/'
    import os
    assert os.path.exists(direct)
    return direct



###################
def check_pairing(i, j, dist_matrix, idx_dict):
    # return the two values that realize bd
    # works for bd only
    # needs to refactor
    import numpy as np
    assert max(i, j) < np.shape(dist_matrix)[0]
    dist = dist_matrix[i, j]
    pts = idx_dict[(i, j)]
    if type(pts) == str:
        assert (pts == 'same') or (pts == 'emas')
        return ('Exact same', 233, 233)

    if pts[0] != None and pts[1] != None:
        # print(pts, dist, i, j)
        if abs(abs(pts[0][0] - pts[1][0]) - dist) < 0.01:
            return ('diff diagrams', pts[0][0], pts[1][0])  # birth
        elif abs(abs(pts[0][1] - pts[1][1]) - dist) < 0.01:
            return ('diff diagrams', pts[0][1], pts[1][1])  # death

    elif pts[0] == None and pts[1] != None:
        assert abs(dist - abs((pts[1][0] - pts[1][1]) / 2)) < 0.01
        return ('same diagram1', pts[1][0], pts[1][1])

    elif pts[0] != None and pts[1] == None:
        assert abs(dist - abs((pts[0][0] - pts[0][1]) / 2)) < 0.01
        return ('same diagram0', pts[0][0], pts[0][1])

    elif pts[0] == None and pts[1] == None:
        return ('overlap', None, None)
    else:
        print ('Unconsidered Cases')
        raise AssertionError
def test_check_pairing(i, dist_matrix, idx_dict):
    import numpy as np
    n = np.shape(dist_matrix)[0]
    for j in range(n):
        try:
            print((i,j,check_pairing(i, j, dist_matrix, idx_dict)))
        except:
            print((i,j, dist_matrix[i,j], idx_dict[(i, j)]))
def get_nodeid(g, key, value, tol=1e-4):
    # in graph g, find the node id whose key-val is value under some tolerance
    n_nodes = len(g)
    idx = list(g.nodes())[0]
    assert key in list(g.nodes[idx].keys())
    assert type(key) == str
    for i in list(g.nodes()):
        if abs(g.nodes[i][key] - value) < tol:
            return i
    # print(g)
    # print('For key=%s, value=%s, Fail to find Node ID'%(key, value))
    raise Nodeid_Unfound_Error('Fail to find Node ID whose value is %s'%value)
def get_nodesid(i, j, graphs, key='fv', debug='off'):
    # for graph i and graph j, find the two node ID(not necessarily belong to different graphs)
    # that realize the bd dist(in dist_matrix, idx_dict)

    def get_data(g, id, debug='off'):
        # sub-dict.
        # for graph g[id], get the filtration value for deg, cc, fv, fiedler, ricci
        if id == None:
            print()
        if debug == 'on':
            print((id), end=' ')
            print((g.nodes[id]))
        return {'deg': g.nodes[id]['deg'], 'cc': g.nodes[id]['cc'], 'label': g.nodes[id]['label'],
                'fv': g.nodes[id]['fv'], 'fiedler':g.nodes[id]['fiedler'], 'ricciCurvature':g.nodes[id]['ricciCurvature']}

    def formatkey(i, id):
        # output i_id, where id is the node id and i is graph id
        assert type(i) == int
        assert type(int(id)) == int
        return str(i) + str('_') + str(id)

    for k1 in range(len(graphs[i])):
        for k2 in range(len(graphs[j])):
            try:
                g1 = graphs[i][k1]
                g2 = graphs[j][k2]
                tmp = check_pairing(i, j, dist_matrix, idx_dict)
                if tmp == None:
                    # implement later
                    print(('Check_pairing_error: i is %s, j is %s, distance is %s, dict is %s'%(i,j,dist_matrix[i,j],idx_dict[(i,j)])))
                    return 'check_pairing_error'
                (t, v1, v2) = check_pairing(i, j, dist_matrix, idx_dict)
                if debug == 'on':
                    print((t,v1,v2))
                if t == 'Exact same':
                    id1 = list(g1.nodes)[0] # id doesn't matter here
                    id2 = list(g2.nodes)[0]
                    return {formatkey(i, id1):  get_data(g1, id1), formatkey(j, id2):  get_data(g2, id2)} #needs to refactor

                if t == 'diff diagrams':
                    id1 = get_nodeid(g1, key, v1)
                    id2 = get_nodeid(g2, key, v2)
                    return {formatkey(i, id1): get_data(g1, id1), formatkey(j, id2): get_data(g2, id2)}

                elif t == 'same diagram0':
                    id1 = get_nodeid(g1, key, v1)
                    id2 = get_nodeid(g1, key, v2)
                    # assert id1!=id2
                    return {formatkey(i, id1): get_data(g1, id1), formatkey(i, id2): get_data(g1, id2)}

                elif t == 'same diagram1':
                    id1 = get_nodeid(g2, key, v1)
                    id2 = get_nodeid(g2, key, v2)
                    if id1 == id2: # ('same diagram0', 1.0, 1.0000100135803223) for protein_data, i = 955, j = 31
                        pass
                    return {formatkey(j, id1): get_data(g2, id1), formatkey(j, id2): get_data(g2, id2)}

                elif t == 'overlap':
                    pass
                else:
                    print('Unconsidered case')
                    raise AssertionError
            except Nodeid_Unfound_Error:
                if (k1 == len(graphs[i])-1) and (k2 == len(graphs[j])-1):
                    print('Truly Trouble')
                    return 'TruelyTrouble'
                else:
                    pass
def gradient(i, j, dist_matrix):
        # compute d k_ij/d beta
        # (tmp, v1, v2) = check_pairing(i, j, dist_matrix, idx_dict)
        import numpy as np
        def graphid(s):
            # s: graphid_nodeid, return the graphid
            assert type(s) == str
            assert s.count('_') == 1
            return s.split('_')[0]

        def clean_key(d):
            # for tmp_data, get the keys that before the '_'
            assert type(d) == dict
            keys = list(d.keys())
            return [graphid(key) for key in keys]
        # clean_key(tmp_data)
        if i == j:
            return np.array([0, 0, 0, 0, 0])
        try:
            tmp_data = get_nodesid(i, j, graphs) # {'1_4': {'cc': 1.0, 'deg': 1}, '1_15': {'cc': 1.8860759493670887, 'deg': 0}}
        except:
            print((i,j))
        if (tmp_data == 'check_pairing_error') or (tmp_data == 'TruelyTrouble'):
            return np.array([0,0,0,0,0])
        # tmp_data = {'1_15': {'cc': 1.8860759493670887, 'deg': 1, 'fiedler': 1.0, 'fv': 1.8860759493670887,'ricciCurvature': 1.0},
        #             '3_21': {'cc': 1.8166666666666667, 'deg': 1, 'fiedler': 1.0, 'fv': 1.8166666666666667, 'ricciCurvature': 0.9999999999999993}}
        try:
            assert len(tmp_data) == 2
        except:
            return np.array([0, 0, 0, 0, 0])
            print((i, j, tmp_data))
            print((check_pairing(i, j, dist_matrix, idx_dict)))
            AssertionError
        keys = clean_key(tmp_data)
        vals = list(tmp_data.values())
        assert len(keys) == 2
        idx0 = int(keys[0]); idx1 = int(keys[1])
        if keys[0] != keys[1]:
            try:
                (dist_matrix[idx0, idx1] - abs(vals[0]['fv'] - vals[1]['fv'])) < 0.01
            except AssertionError:
                print((i, j, dist_matrix[idx0, idx1], abs(vals[0]['fv'], vals[1]['fv'])))
                print((dist_matrix[idx0, idx1] - abs(vals[0]['fv'] - vals[1]['fv'])))
        elif keys[0] == keys[1]:
            assert (dist_matrix[idx0, idx1] - abs(vals[0]['fv'] - vals[1]['fv'])/2.0) < 0.01

        if (vals[0]['fv'] - vals[1]['fv']) > 0:
            big = 0
            small = 1
        else:
            big = 1
            small = 0
        vec = {}
        for func in ['deg', 'ricciCurvature', 'fiedler', 'cc']:
            vec[func] = vals[big][func] - vals[small][func]

        dist_ij = dist_matrix[i, j]
        constant = (np.e ** (- dist_ij * dist_ij)/float(sigma)) * (-2 * dist_ij) * (1/float(sigma)) # important
        return constant * np.array([vec['deg'], vec['ricciCurvature'], vec['fiedler'], vec['cc'], 0])
        # return constant * np.array([vec['deg'], 0, 0, vec['cc'], 0])
def back_prop(cache, sigma, i = 1 ):
    # square loss(hopefully same with svr)
    import numpy as np
    (k, y, y_hat, beta, alpha) = cache # k is not used here
    assert len(beta)==5
    d_beta = np.zeros(np.shape(beta))
    n = np.shape(alpha)[0]
    assert np.shape(alpha) == (n,)

    # tmp = 2* (y[i] - y_hat[i])
    tmp = hinge_gradient(y[i], y_hat[i])
    if tmp == 0:
        return np.array([0, 0, 0, 0, 0])
    # gradient(1,2, dist_matrix)
    tmp2_i = np.zeros((n, 5))
    for j in range(n):
        # print(j),
        tmp2_i[j] = gradient(i, j, dist_matrix)
    assert tmp2_i[j][4] == 0
    gd_i = np.dot(alpha, tmp2_i) * tmp
    # print ('gd_i is %s'%gd_i)
    return gd_i
def svm_hyperparameter(dist_matrix, Y, n_jobs=-1, loss_type='hinge', hyperparameter_flag='no'):
    # grid search the c and sigma for svm
    if hyperparameter_flag == 'yes':
        # input: dist_matrix
        # grid search hyperparameter for kernel induced by bd_kernel
        threshold = 0
        scores = []
        for sigma in [0.1, 1, 10, 100]:
            print(('Sigma is %s' % sigma))
            kernel = np.exp(np.multiply(dist_matrix, -dist_matrix) / sigma)
            for c in [0.01, 0.1, 1, 10, 100]:
                # for c in [1]:
                svm_data = Parallel(n_jobs=n_jobs)(
                    delayed(single_test)(kernel, c, Y, r_seed, lbd=1e-8, loss=loss_type, debug='off') for r_seed in
                    range(n_runs))
                score = get_accuracies(svm_data, c, sigma, threshold, loss=loss_type)
                scores = scores + [score]
            # null = Parallel(n_jobs=n_jobs, verbose=10)(delayed(test_bdkernel)(kernel, c, Y, loss='hinge') for c in [.1,1,10])
            # null = Parallel(n_jobs=n_jobs)(delayed(test_bdkernel)(kernel, 1, Y, ldb, loss='square') for ldb in [1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
            print('\n')
        print('Finish svm and hyperparameter search')
        best8 = [(score['c'], score['sigma'], score['train_acc'], score['test_acc'], score['test_std']) for score in
                 sorted(scores, key=lambda a: a['test_acc'], reverse=True)[0:8]]
        print_best(best8)
        return best8
def total_gradient(cache, idx, sigma, debug='off'):
    import time
    start = time.time()
    from joblib import delayed, Parallel
    total = np.zeros((1,5))+1.111
    total = Parallel(n_jobs=n_jobs)(delayed(back_prop)(cache, sigma, i) for i in idx)
    total = sum(total)

    if debug == 'on':
        print((type(total)), end=' ')
        print((total), end=' ')
        print(('the shape of total before sum is %s' % str(np.shape(total))), end=' ')
        # total = np.zeros((1,5))
        # for i in idx:
        #     if debug == 'on':
        #         print(i),
        #     total = total + back_prop(cache, sigma, i)
        # print('Serial Total is %s' % total)
        # normalize total
        total = total / np.linalg.norm(total)
        print(('the shape of total after sum is %s' %str(np.shape(total))))
        print(total)
        assert total[4]==0
    print(('Computing total gradient takes %s .'%(time.time()-start)), end=' ')
    return total
def check_trainloss(check_cache):
    import numpy as np
    from sklearn.metrics import accuracy_score
    (old_svm_data, svm_data) = check_cache
    for keys in ['train_idx', 'test_idx']:
        try:
            assert old_svm_data[0][keys] == svm_data[0][keys]
        except AttributeError:
            assert (old_svm_data[0][keys] == svm_data[0][keys]).all()

    old_kernel = old_svm_data[0]['kernel']; old_clf = old_svm_data[0]['clf']; kenel_train = svm_data[0]['kernel']
    y = old_svm_data[0]['y']
    train_idx = old_svm_data[0]['train_idx']; test_idx = old_svm_data[0]['test_idx']

    old_kernel_train = old_kernel[np.ix_(train_idx, train_idx)]
    old_kernel_test = old_kernel[np.ix_(test_idx, test_idx)]
    kernel_train = kernel[np.ix_(train_idx, train_idx)]

    assert accuracy_score(np.sign(old_clf.predict(old_kernel_train)), y[np.ix_(train_idx)]) == accuracy_score(np.sign(old_svm_data[0]['y_hat'][train_idx]), y[np.ix_(train_idx)])
    print(('Accuracy of old clf on traning set is %s .' % accuracy_score(np.sign(old_clf.predict(old_kernel_train)), y[train_idx])), end=' ')
    print(('Accuracy of new clf on traning set is %s .' % accuracy_score(np.sign(old_clf.predict(kernel_train)), y[train_idx])))
    return
def total_loss(itr, cache, train_idx):
    (k, y, y_hat, beta, alpha) = cache  # k is not used here
    total_loss = 0; n_train = len(y); all_loss= [0] * n_train;
    test_loss = 0; all_test_loss = [0] * (len(y) - n_train)
    for i in train_idx:
        total_loss += hinge_loss(y[i], y_hat[i])
        all_loss[i] = hinge_loss(y[i], y_hat[i])

    print(('The total loss on dataset of size %s is %s .'%(len(train_idx), total_loss)), end=' ')
    # print(all_loss)
    import matplotlib.pyplot as plt
    ax = plt.subplot()
    ax.plot(all_loss, 'ro:', markersize=1)
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/gd/loss_dist/'
    make_direct(direct)
    filename = str(itr) + '_' + str(beta) + '.png'
    plt.title('Iteration: %s,  Total loss: %s'%(itr, np.round(total_loss)))
    plt.savefig(direct + filename)
    plt.close()
    return total_loss
def node_baseline_variants():
    if True:
        # variants of baseline
        print((list(graphs_[0][0].node[0].keys())))
        allowed = ['deg', '1_0_deg_min', '1_0_deg_mean', '1_0_deg_std', '1_0_deg_max', '2_deg_min', '2_deg_max',
                   '3_deg_min', '3_deg_max']
        allowed = ['1_1_label', '1_1_deg', 'deg', '1_0_deg_min', '1_0_deg_mean', '1_0_deg_std', '1_0_deg_max',
                   '1_0_label_min', '1_0_label_max', '1_0_label_mean']
        allowed = ['deg', '1_0_deg_min', '1_0_deg_mean', '1_0_deg_std', '1_0_deg_max', '1_1_deg', 'ricciCurvature',
                   '1_1_ricciCurvature']
        allowed = ['deg', '1_0_deg_min', '1_0_deg_mean', '1_0_deg_std', '1_0_deg_max', '1_1_deg', 'label']
        X = merge_features(graphs_, allowed, 30)
        X = normalize_(X)
        rfclf(X, Y, m_f=40)
        max(np.linalg.norm(X, axis=1)) == 1 or max(np.linalg.norm(X, axis=0)) == 1
        if graph == 'reddit_5K' or graph == 'reddit_12K':
            nonlinear_flag = 'False'

            param = {'kernel': 'linear', 'C': 1000}
        else:
            nonlinear_flag = 'True'
            param = searchclf(1001, test_size=0.1, nonlinear_flag=nonlinear_flag, verbose=1)
        evaluate_clf(graph, X, Y, param, n_splits=10)


def dignoise_kernel(true_kernel, fake_kernel):
    import scipy
    print(('The difference bwn true kernel and fake kernel is %s' % (true_kernel - fake_kernel)))
    print(('norm of true kernel is %s' % scipy.linalg.norm(true_kernel)))
    print(('norm of fake kernel is %s' % scipy.linalg.norm(fake_kernel)))
    print(('norm of difference is %s' % scipy.linalg.norm(fake_kernel - true_kernel, 'fro'), 'fro'))
    print(('norm of difference is %s' % scipy.linalg.norm(fake_kernel - true_kernel, 1), 1))
    print(('norm of difference is %s' % scipy.linalg.norm(fake_kernel - true_kernel, -1), -1))
    print(('norm of difference is %s' % scipy.linalg.norm(fake_kernel - true_kernel, 2), 2))
    print(('norm of difference is %s' % scipy.linalg.norm(fake_kernel - true_kernel, -2), -2))


################
# some old functions
######################


######################
# from joblib import Parallel, delayed
# results = Parallel(n_jobs=n_jobs)(delayed(pipeline1)(mutag, i) for i in range(1, 5))
# class others():
#     def compute_bd(dgm1, dgm2):
#         import gudhi
#         def dgm2diag(dgm):
#             diag = list()
#             for pt in dgm:
#                 if str(pt.death) == 'inf':
#                     diag.append([pt.birth, float('Inf')])
#                 else:
#                     diag.append([pt.birth, pt.death])
#             print(diag)
#             return diag
#
#         if str(type(dgm1)) == str(type(dgm2)) == "<class 'dionysus._dionysus.Diagram'>":
#             diag1 = dgm2diag(dgm1)
#             diag2 = dgm2diag(dgm2)
#         elif type(dgm1) == type(dgm2) == list:
#             diag1 = dgm1
#             diag2 = dgm2
#         return gudhi.bottleneck_distance(diag1, diag2)
#
#     def edit_array(i):
#         return arr[i] + ' hello!'
#
#     def example(arr):
#         return 0
#         import numpy as np
#         from multiprocessing import Pool
#         if True:
#             pool = Pool(processes=2)
#             list_start_vals = range(len(arr))
#             array_2D = pool.map(edit_array, list_start_vals)
#             pool.close()
#             print array_2D
#
#     arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
#     example(arr)
#     import numpy as np
#     from multiprocessing import Pool
#
#     arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
#
#     def edit_array(i):
#         return arr[i] + ' hello!'
#
#     if True:
#         pool = Pool(processes=2)
#         list_start_vals = range(len(arr))
#         array_2D = pool.map(edit_array, list_start_vals)
#         pool.close()
#         print array_2D
def test_diag2dgm():
    diag1 = [[9.5, 14.1], [2.8, 4.45], [3.2, 10]]
    diag2 = [[0, 1]]
    dgm1 = diag2dgm(diag1)
    dgm2 = diag2dgm(diag2)
    import dionysus as d
    d.bottleneck_distance(dgm1, dgm2)
def parallel(graphs):
    dgms = [0] * 200
    for i in range(len(graphs)):
        dgm = get_diagram(graphs[i])
        dgms[i] = dgm
    return dgms
def bd_matrix(data):
    def idgm(data, i):
        g = convert2nx(data[i], i)
        g = graph_processing(g)
        dgm = get_diagram(g)
        return dgm

    def d_compute_bd(dgm1, dgm2):
        import dionysus as d
        bdist = d.bottleneck_distance(dgm1[1], dgm2[1])
        return bdist

    from joblib import Parallel, delayed
    n = len(data)
    d_mtx = np.zeros((n, n))
    dgm_list = [0] * n
    # Parallel(n_jobs=n_jobs)(delayed(idgm)(data, i) for i in range(n))
    for i in range(n):
        dgm_list[i] = idgm(data, i)
    return dgm_list


import cProfile
def profile_this(fn):
    def profiled_fn(*args, **kwargs):
        # name for profile dump
        prof = cProfile.Profile()
        ret = prof.runcall(fn, *args, **kwargs)
        print(ret)
        return ret
    return profiled_fn

def test():
    x = list(range(10)) + list(range(100))
    return 2 + 2

test()

from profilehooks import profile
class SampleClass:
    # @profile
    def silly_fibonacci_example(self, n):
        if n < 1:
            raise ValueError('n must be >= 1, got %s' % n)
        if n in (1, 2):
            return 1
        else:
            return (self.silly_fibonacci_example(n - 1) +
                    self.silly_fibonacci_example(n - 2))

from profilehooks import coverage

def silly_factorial_example(n):
    """Return the factorial of n."""
    if n < 1:
        raise ValueError('n must be >= 1, got %s' % n)
    if n == 1:
        return 1
    else:
        return silly_factorial_example(n - 1) * n
def del_inf(dgms):
    import dionysus as d
    dgms_list = [[], []]
    for i in range(2):
        pt_list = list()
        for pt in dgms[i]:
            if (pt.birth == float('inf')) or (pt.death == float('inf')):
                pass
            else:
                pt_list.append(tuple([pt.birth, pt.death]))
        # pt_list.append(tuple([0, 0])) # append a dummy point
        diagram = d.Diagram(pt_list)
        dgms_list[i] = diagram
    return dgms_list

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

def convert2nx(graph, i, print_flag='False'):
    # graph: python dict
    import networkx as nx
    keys = graph.keys()
    try:
        assert keys == range(len(graph.keys()))
    except AssertionError:
        # pass
        print('%s graph has non consecutive keys'%i)
        print('Missing nodes are the follwing:')
        for i in range(max(graph.keys())):
            if i not in graph.keys():
                print(i),
        # print('$') # modification for reddit binary
    gi = nx.Graph()
    # add nodes
    for i in keys:
        gi.add_node(i) # change from 1 to i. Something wired here

    assert len(gi) == len(keys)
    # add edges
    for i in keys:
        for j in graph[i]['neighbors']:
            if j > i:
                gi.add_edge(i, j)
    # add labels
    # print(gi.node)
    for i in keys:
        # print graph[i]['label']
        if graph[i]['label']=='':
            gi.node[i]['label'] = 1
            # continue
        try:
            gi.node[i]['label'] = graph[i]['label'][0]
        except TypeError: # modifications for reddit_binary
            gi.node[i]['label'] = graph[i]['label']
        except IndexError:
            gi.node[i]['label'] = 0 # modification for imdb_binary
    # print(gi)
    # print (gi.vs[1])
    assert len(gi.node) == len(graph.keys())
    gi.remove_edges_from(gi.selfloop_edges())
    if print_flag=='True':
        print('graph: %s, n_nodes: %s, n_edges: %s' %(i, len(gi), len(gi.edges)) )

    return gi
if __name__ == '__main__':
    silly_factorial_example(10)

if __name__ == '__main__':
    fib = SampleClass().silly_fibonacci_example
    print(fib(10))