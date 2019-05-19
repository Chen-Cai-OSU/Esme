import dionysus as d
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.preprocessing import normalize

from .tools import make_direct, dgm2diag, remove_zero_col, normalize_dgm, flip_dgm

def function_basis(g, key='dist', simplify_flag = 0):
    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    import time
    start = time.time()
    import networkx as nx
    import sys
    import numpy as np
    from helper import attribute_mean

    assert nx.is_connected(g)
    assert nx.is_tree(g)
    sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/')
    from GraphRicciCurvature.OllivierRicci import ricciCurvature
    def norm(g, key):
        # return 0.01
        # get the max of g.node[i][key]
        for v, data in sorted(g.nodes(data=True), key=lambda x: abs(x[1][key]), reverse=True):
            norm = np.float(data[key])
            return norm

    g_ricci = g

    # degree
    deg_dict = dict(nx.degree(g))
    deg_norm = np.float(max(deg_dict.values()))
    for n in g_ricci.nodes():
        g_ricci.node[n]['deg'] = deg_dict[n]
    for n in g_ricci.nodes():
        attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=0)

    # hop
    dist_dict = nx.single_source_dijkstra_path_length(g_ricci, 1, weight='length')
    for n in g_ricci.nodes():
        g_ricci.node[n]['dist'] = dist_dict[n]
        g_ricci.node[n]['direct_distance'] = np.linalg.norm(np.array(g.nodes[n]['coordinates']) - np.array(g.nodes[1]['coordinates']) )
    for n in g_ricci.nodes():
        attribute_mean(g_ricci, n, key='direct_distance', cutoff=1, iteration=0)

    # cc
    closeness_centrality = nx.closeness_centrality(g)  # dict
    closeness_centrality = {k: v / min(closeness_centrality.values()) for k, v in closeness_centrality.iteritems()} # no normalization for debug use
    closeness_centrality = {k: 1.0 / v for k, v in closeness_centrality.iteritems()}
    for n in g_ricci.nodes():
        g_ricci.node[n]['cc'] = closeness_centrality[n]

    for n in g_ricci.nodes():
        g_ricci.node[n]['x'] = g_ricci.node[n]['coordinates'][0]
        g_ricci.node[n]['y'] = g_ricci.node[n]['coordinates'][1]
        g_ricci.node[n]['z'] = g_ricci.node[n]['coordinates'][2]

    if simplify_flag == 1:
        # fiedler
        from networkx.linalg.algebraicconnectivity import fiedler_vector
        fiedler = fiedler_vector(g, normalized=False)  # np.ndarray
        assert max(fiedler) > 0
        fiedler = fiedler / max(fiedler)
        assert max(fiedler) == 1
        for n in g_ricci.nodes():
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['fiedler'] = fiedler[n_idx]


        # closeness_centrality
        closeness_centrality = nx.closeness_centrality(g)  # dict
        closeness_centrality = {k: v / min(closeness_centrality.values()) for k, v in
                                closeness_centrality.iteritems()}  # no normalization for debug use
        closeness_centrality = {k: 1.0 / v for k, v in closeness_centrality.iteritems()}
        for n in g_ricci.nodes():
            g_ricci.node[n]['cc'] = closeness_centrality[n]

        try:
            g_ricci = ricciCurvature(g, alpha=0.5, weight='length')
            assert g_ricci.node.keys() == list(g.nodes())
            ricci_norm = norm(g, 'ricciCurvature')
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] /= ricci_norm
        except:
            print('RicciCurvature Error for graph, set 0 for all nodes')
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] /= ricci_norm

    # print('Graph of size %s, it takes %s'%(len(g), time.time() - start))
    return g_ricci
def set_label(new_neuron_flag=False, sample_flag=False):
    if new_neuron_flag == False:
        Y = np.array([1] * 710 + [7] * 420).reshape(1130, )
        n = 1268
    elif new_neuron_flag == True and sample_flag == False:
        n = 911
        Y = np.array([[1] * 147 + [2] * 17 + [3] * 123 + [4] * 94 + [2] * 78 + [5] * 452]).reshape(n, )
    elif new_neuron_flag == True and sample_flag == True:
        n = 559 - 100  # remove last class
        # Y = np.array([[1] * 147 + [2] * 17 + [3] * 123 + [4] * 94 + [2] * 78 + [5] * 100]).reshape(n, )
        Y = np.array([[1] * 147 + [2] * 17 + [3] * 123 + [4] * 94 + [2] * 78]).reshape(n, )
    return (n, Y)

def distance(i,j, treedf):
    # eculidean distance of two nodes
    import numpy as np
    df = treedf
    coord1 = np.array([df['x'][i], df['y'][i], df['z'][i]])
    coord2 = np.array([df['x'][j], df['y'][j], df['z'][j]])
    dist = np.linalg.norm(coord1-coord2)
    return dist

def convert2nx(df):
    # graph: python dict
    import networkx as nx
    gi = nx.Graph()
    n = len(df)
    for i in range(1,n+1): # change later
        gi.add_node(i, coordinates = (df['x'][i], df['y'][i], df['z'][i]), structure = df['structure'][i], radius = df['radius'][i], parent = df['parent'][i])
        if df['parent'][i] != -1:
            gi.add_edge(int(df['id'][i]), int(df['parent'][i]), length = distance(i, df['parent'][i], df))
    assert nx.is_tree(gi)
    # assert gi.nodes[1]['coordinates'] == (0, 0, 0)
    return gi

def distance_(i, j, tree):
    # eculidean distance of two nodes
    import numpy as np
    coord1 = np.array(tree.nodes[i]['coordinates'])
    coord2 = np.array(tree.nodes[j]['coordinates'])
    dist = np.linalg.norm(coord1 - coord2)
    return dist

def print_node_vals(g, key):
    n = len(g)
    for i in range(1, n+1):
        print(i, g.nodes[i][key])

def find_node_val(g, key, val):
    import numpy as np
    n = len(g); flag = 0
    for i in range(1, n + 1):
        if np.abs(g.nodes[i][key] - val) < 0.01:
            print(i, g.nodes[i][key]);
            flag = 1
    if flag == 0:
        print('Did not match')

def number_lines(i, files):
    file_directory = files[i - 1]
    count = len(open(file_directory).readlines())
    return count

def count_lines():
    for i in range(1, 1000):
        print(i,  number_lines(i))

def dgm_distinct(dgm):
    diag = dgm2diag(dgm)
    diag = dgm2diag(dgm)
    distinct_list = [i[0] for i in diag]
    distinct_list += [i[1] for i in diag]
    distinct_list.sort()
    return distinct_list
def convert2dayu(g):
    direct = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Test/dayu.txt'
    f= open(direct, 'w')
    f.write(str(len(g)) + '\n' )
    for i in g.nodes():
        nval = g.nodes[i]['direct_distance']
        f.write(str(nval) + '\n')
    for e in g.edges():
        f.write(str(e[0]-1) + ' ' + str(e[1]-1) + '\n' )
        # print e[0],
        # print e[1]
def dgm_uptriangle(dgm):
    for p in dgm:
        assert p.death >= p.birth
def export_dgm(i, dgm, files, key='null'):
    file1 = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/TruePDs/' + key + '/'
    file2 = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/TruePDs/' + key + '/'
    neuron_id = str(i) + '_' + files[i-1].split('/')[-1]
    try:
        make_direct(file1)
        f = open(file1 + neuron_id, 'w+')
    except:
        make_direct(file2)
        f = open(file2 + neuron_id, 'w+')
    diag = dgm2diag(dgm)
    for pd in diag:
        f.write(str(pd[0]) + ' ' + str(pd[1]) + '\n')
    f.close()
def average_dgm_pts(dgm):
    assert len(dgm) > 0
    dgm = normalize_dgm(dgm)
    if dgm[0].birth >= dgm[0].death:
        dgm = flip_dgm(dgm)
    for p in dgm:
        assert p.death <= 1.001
    n = len(dgm)
    weights = [p.death - p.birth for p in dgm]
    birth_ave = np.average([p.birth for p in dgm])
    birth_weighted_ave = np.average([p.birth for p in dgm], weights=weights)
    birth_max = np.max([p.birth for p in dgm])
    birth_min = np.min([p.birth for p in dgm])
    birth_std = np.std([p.birth for p in dgm])

    death_ave = np.average([p.death for p in dgm])
    death_weighed_ave = np.average([p.death for p in dgm], weights=weights)
    lifetime_ave = np.average([p.death-p.birth for p in dgm])
    death_max = np.max([p.death for p in dgm])
    death_min = np.min([p.death for p in dgm])
    death_std = np.std([p.death for p in dgm])

    from scipy.stats import gmean
    birth_product = gmean([p.birth+0.001 for p in dgm])
    death_product = gmean([p.death + 0.001 for p in dgm])
    if np.isnan(birth_product) or np.isnan(death_product):
        birth_product = death_product = 0
    return np.array([birth_ave, death_ave, lifetime_ave, birth_product, death_product, birth_min, birth_max, death_min, death_max, birth_std, death_std])
    return np.array([birth_ave, death_ave, lifetime_ave, birth_product, death_product, birth_weighted_ave, death_weighed_ave])
def multihistogram(dgm, n_bin = 20, range_ = (0,1)):
    assert len(dgm)>0
    if dgm[0].birth >= dgm[0].death:
        dgm = flip_dgm(dgm)
    dgm_uptriangle(dgm)
    dgm = normalize_dgm(dgm)
    dgmx = [p.birth for p in dgm]
    dgmy = [p.death for p in dgm]
    try:
        assert len(dgmx) == len(dgmy)
    except AssertionError:
        print ('dgmx is %s, and dgmy is %s'%(dgmx, dgmy))
    assert max(max(dgmx), max(dgmy)) < 1.01
    xedges = np.linspace(range_[0], range_[1], n_bin + 1)
    yedges = np.linspace(range_[0], range_[1], n_bin + 1)
    H, xedges, yedges = np.histogram2d(dgmx, dgmy, bins=(xedges, yedges))
    assert np.sum(H) == len(dgmx)
    return H.reshape(1, n_bin**2)
def pairing_feature(dgms, n_bin=20, range_=(0,1), cor_flag=False):
    pairing = np.zeros((len(dgms), n_bin**2))
    for i in range(len(dgms)):
        pairing[i] = multihistogram(dgms[i], range_=range_, n_bin=n_bin)
    (pairing, cor_dict, inv_cor_dict) = remove_zero_col(pairing, cor_flag=cor_flag)
    pairing = normalize(pairing, axis=1)
    print ('The shape of pairing feature is, ', np.shape(pairing))
    if cor_flag==True:
        return (pairing, cor_dict, inv_cor_dict)
    else:
        return pairing
def aggregation_feature(dgms):
    feature = np.zeros((len(dgms), 9+2))
    for i in range(len(dgms)):
        feature[i] = average_dgm_pts(dgms[i])
    feature = normalize(feature)
    return feature
def int_dgm(dgm):
    # only use before normaliztion of dgm
    dgm_int = []
    for p in dgm:
        dgm_int.append((int(p.birth), int(p.death)))
    return d.Diagram(dgm_int)


def scatter_comp():
    import matplotlib.pyplot as plt
    import numpy as np
    data = dgm2diag(Neuron_dgm)
    x = np.array([data[i][0] for i in range(len(data)-1)])
    y = np.array([data[i][1] for i in range(len(data)-1)])
    plt.scatter(x, y, c='b')
    plt.show()


def get_dict_key(val_dict, value):
    FirstIndex = lambda a, val, tol: next(i for i, _ in enumerate(a) if np.isclose(_, val, tol))
    a = val_dict.values()
    idx = FirstIndex(a, value, tol=1e-6)
    start_key = min(val_dict.keys())
    if abs(val_dict[idx + start_key] - value) > 0.01:
        print('Looing for %s, idx is %s, found %s' % (value, idx, val_dict[idx]))
    return val_dict.keys()[idx]
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def get_coordinates(g, v):
    return np.array((g.node[v]['x'], g.node[v]['y'], g.node[v]['z']))
def angle(g, v, v1, v2):
    # v1 = 112; v2 = 6; v = 4;
    v_coordinates = get_coordinates(g, v)
    v1_coordinates = get_coordinates(g, v1)
    v2_coordinates = get_coordinates(g, v2)
    return angle_between(- v_coordinates + v1_coordinates, -v_coordinates + v2_coordinates)
def get_angle(g, dist_dict, i):
        nbr = list(g.neighbors(i))
        n_nbrs = len(nbr)
        if n_nbrs <= 1:
            return 0.00123
            return 'Only has ' + str(n_nbrs) + ' nbrs'
        elif n_nbrs > 1:
            nbr_dist_dict = {}
            for v in nbr:
                nbr_dist_dict[v] = dist_dict[v]
            parent_dist = min(nbr_dist_dict.values())

            for v in nbr:
                if abs(nbr_dist_dict[v] - parent_dist) < 0.01:
                    parent_dist_key = v
            assert (nbr_dist_dict[parent_dist_key] - parent_dist) < 0.1
            othernodes = [v for v in nbr if v != parent_dist_key]
            try:
                assert len(othernodes) == 2
            except:
                if (i != 0) and (i != 1):
                    print ('Length of nodes %s is not 2' % i),
                    print (othernodes)
            # print(i, othernodes)
            angle_ = angle(g, i, othernodes[0], othernodes[1])
            return angle_


def str_feature(structure_dgm):
    feature = np.zeros(16)
    # str_death = [p[0] for p in structure_dgm]
    # str_birth = [p[1] for p in structure_dgm]
    c = Counter(structure_dgm)
    n = sum(c.values())
    for key, val in c.items():
        idx = int((key[0] - 1) * 4 + (key[1] - 1))
        feature[idx] = val / float(n)
    return feature
def stat(lis, high_order=False):
    lis = [a for a in lis if a!=0.00123]
    # list = angledgm
    if high_order == True:
        pass
    return np.array([np.min(lis), np.max(lis), np.median(lis), np.mean(lis), np.std(lis)])
def homfeature(i, dgm, key = 'dist'):
    # output the location of homology feature for tree i, node v
    # i = 10
    # dgm = unnormalized_dgms[i]
    assert 'trees' in globals().keys()
    tree = trees[i]
    tree = function_basis(tree, key='deg')
    tree = function_basis(tree, key='cc')
    g = function_basis(tree, key=key)
    # value = pt.birth # value = 82.87595916651989
    # dgm = get_diagram(g, key=key) # not sure here

    # value = 65.59537941044323
    val_dict = nx.get_node_attributes(g, key)
    # node_v = get_dict_key(val_dict, value)

    value_list =  [[p.birth, p.death] for p in dgm]
    dist_dict = nx.get_node_attributes(g, key)
    nodeparing_list = [(get_dict_key(val_dict, val[0]), get_dict_key(val_dict, val[1])) for val in value_list]

    # for i in range(15, 100):
    #     angle_list = []
    #     for v in g.nodes():
    #         angle_list.append(get_angle(g, dist_dict, v))
    #     try:
    #         assert abs(len(angle_list) - angle_list.count(0.00123)*2) < 10
    #     except:
    #         print('tree %s has discrepence %s'%(i, abs(len(angle_list) - angle_list.count(0.00123)*2)/np.float(len(angle_list))))
    def node_i_feature(g, v, dist_dict, angle_only_flag = False):
        dict = g.node[v]
        (x,y,z, strucutre) = (dict['x'], dict['y'], dict['z'], dict['structure'])
        radius = dict['radius']
        angle = get_angle(g, dist_dict, v)
        degree = dict['deg']
        cc = dict['cc']
        if angle_only_flag:
            return angle
        return (x, y, z, strucutre, radius, angle, degree, cc)

    angle_list = []
    for pairnode in nodeparing_list:
        angle_list.append((node_i_feature(g, pairnode[0], dist_dict, angle_only_flag=False), node_i_feature(g, pairnode[1], dist_dict, angle_only_flag=False)))
    return angle_list
def dgm_decoration(i):
    argument_dgm = homfeature(i, unnormalized_dgms[i], key=filtration_type)
    structure_dgm =[(p[0][3], p[1][3]) for p in argument_dgm]
    feature1 = str_feature(structure_dgm)

    position_dgm = [(p[0][0:3], p[1][0:3]) for p in argument_dgm]

    radius_dgm = [(p[0][4], p[1][4]) for p in argument_dgm]
    radius_birth = [p[1] for p in radius_dgm]
    radius_death = [p[0] for p in radius_dgm]
    feature2 = np.concatenate((stat(radius_birth), stat(radius_death)))

    angledgm = [p[0][5] for p in argument_dgm]
    feature3 = stat(angledgm)
    if i % 50 ==0:
        print ('*'),

    deg_dgm = [(p[0][6], p[1][6]) for p in argument_dgm] # deg
    deg_birth = [p[1] for p in deg_dgm]
    deg_death = [p[0] for p in deg_dgm]
    feature4 = np.concatenate((stat(deg_birth), stat(deg_death)))

    cc_dgm = [(p[0][7], p[1][7]) for p in argument_dgm]  # cc
    cc_birth = [p[1] for p in cc_dgm]
    cc_death = [p[0] for p in cc_dgm]
    feature5 = np.concatenate((stat(cc_birth), stat(cc_death)))

    return np.concatenate((feature1, feature2, feature3, feature4, feature5))
def viz_tree(i, show_flag=False):

        assert 'trees' in globals().keys()
        assert 'Y' in globals().keys()
        tree = trees[i]
        try:
            import pygraphviz
            from networkx.drawing.nx_agraph import graphviz_layout
        except ImportError:
            try:
                import pydot
                from networkx.drawing.nx_pydot import graphviz_layout
            except ImportError:
                raise ImportError("This example needs Graphviz and either "
                                  "PyGraphviz or pydot")

        pos = graphviz_layout(tree, prog='twopi', args='')
        nx.draw_networkx(tree, pos=pos, node_size=5, with_labels=False, node_color = nx.get_node_attributes(tree, 'structure').values())
        import matplotlib.pyplot as plt
        plt.title('Label:%s'%Y[i])
        plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/viz_tree/new_tree/' + str(i) + '.png')
        if show_flag:
            plt.show()
        plt.close()
        if i % 50 == 0:
            print ('#'),


def reshapeX(X_, n):
    if np.shape(X_)[0] != 911:
        X = np.vstack(X_)
    if n == 1268:
        try:
            X = np.concatenate((X_[1 - 1:711, :], X_[849:1269, :]), axis=0)
        except:
            X = np.concatenate((X_[1 - 1:711], X_[849:1269]), axis=0)
    return X
def reg_process(X,n):
    # reshape, remove zero column, and normalize
    X = reshapeX(X,n)
    X = remove_zero_col(X, cor_flag=False)
    X = normalize(X, norm='l2', axis=1, copy=True)
    return X
def split_tree(tree):
    # split the neuron tree by structure type
    # tree = trees[100]
    structure_dict = nx.get_node_attributes(tree, 'structure')
    # print set(structure_dict.values())

    subtree_idx = [key for key, val in structure_dict.items() if val == 1.0]
    subtree_1 = nx.subgraph(tree, subtree_idx)

    subtree_idx = [key for key, val in structure_dict.items() if val == 2.0]
    subtree_2 = nx.subgraph(tree, subtree_idx)

    subtree_idx = [key for key, val in structure_dict.items() if val == 3.0]
    subtree_3 = nx.subgraph(tree, subtree_idx)

    subtree_idx = [key for key, val in structure_dict.items() if val == 4.0]
    subtree_4 = nx.subgraph(tree, subtree_idx)

    return {'1': subtree_1, '2': subtree_2, '3': subtree_3, '4': subtree_4}

def searchclf(X, Y, i, test_size = 0.1, print_flag=False, linear_flag=False, weight_flag=False):
    import time
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn import svm
    from sklearn.metrics import classification_report
    tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                            {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10,100], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    if linear_flag==True:
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=i)
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)
        if print_flag==True:
            print("Best parameters set found on development set is \n %s with score %s" % (
            clf.best_params_, clf.best_score_))
            print(clf.best_params_)
            print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        if print_flag==True:
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print("Detailed classification report:\n")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
        y_true, y_pred = y_test, clf.predict(X_test)
        if print_flag==True:
            print(classification_report(y_true, y_pred))
    if weight_flag:
        clf_ = svm.SVC(kernel='linear', C=1)
        clf_.fit(X,Y) # may need to change to X_train, y_train
        assert np.shape(X)[1] == np.shape(clf_._get_coef())[1]
        print('The weight coefficient is of shape', np.shape(clf_._get_coef()))
        return (clf.best_params_, clf_._get_coef())

    return clf.best_params_
def evaluate_clf(X, Y, best_params_, n_splits):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn import svm
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + 'neuron' + '/result/'
    accuracy = []
    n = 5
    for i in range(n):
        # after grid search, the best parameter is {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
        if best_params_['kernel'] == 'linear':
            clf = svm.SVC(kernel='linear', C= best_params_['C'])
        elif best_params_['kernel'] == 'rbf':
            clf = svm.SVC(kernel='rbf', C=best_params_['C'], gamma=best_params_['gamma'])
        else:
            raise Exception('Parameter Error')
        from sklearn.model_selection import cross_val_score
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
        cvs = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold)
        print(cvs)
        acc = cvs.mean()
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    print('mean is %s, std is %s '%(accuracy.mean(), accuracy.std()))



def test_feat(X, Y, t, linear_flag=False, param_print=False, weight_flag=False):
    if weight_flag == False:
        param = searchclf(X, Y, 1002, test_size=t, linear_flag=linear_flag, weight_flag=False)
    else:
        (param, weight) = searchclf(X, Y, 1002, test_size=t, linear_flag=linear_flag, weight_flag=True)
    evaluate_clf(X, Y, param, n_splits=10)
    if param_print:
        print param

def data_interface(dgm, dynamic_range_flag=True):
    # from dgm to data/max/min
    from aux.tools import assert_dgm_above, print_dgm
    assert_dgm_above(dgm)
    for p in dgm:
        try:
            assert p.death >= p.birth
        except AssertionError:
            print ('birth is %s, death is %s'%(p.birth, p.death))
            raise Exception
    data = [tuple(i) for i in dgm2diag(dgm)]
    try:
        [list1, list2] = zip(*data);
    except:
        print('Problem')
        list1 = [0];
        list2 = [1e-5]  # adds a dummy 0

    if dynamic_range_flag == True:
        min_ = min(min(list1), min(list2))
        max_ = max(max(list1), max(list2))
        std_ = (np.std(list1) + np.std(list2)) / 2.0
    elif dynamic_range_flag == False:
        min_ = -5
        max_ = 5
        std_ = 3

    return {'data': data, 'max': max_ + std_, 'min': min_ - std_}
def rotate_data(data, super_check):
    """
    :param data:
    :return: a list of tuples
    """

    def rotate(x, y):
        return np.sqrt(2) / 2 * np.array([x + y, -x + y])

    def flip(x, y):
        assert x >= y
        return np.array([y, x])

    length = len(data)
    rotated = []

    for i in range(0, length, 1):
        if super_check == True: data[i] = flip(data[i][0], data[i][1])
        point = rotate(data[i][0], data[i][1]);
        point = (point[0], point[1])
        rotated.append(point)
    return rotated
def draw_data(data, imax, imin, discrete_num=500):
    """
    :param data: a list of tuples
    :return: a dictionary: vector of length 1000

    """
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2) + 0.000001))

    discrete_num = discrete_num
    assert imax >= imin
    distr = np.array([0] * discrete_num)
    par = data
    for x, y in par:
        mu = x;
        sigma = y / 3.0
        distr = distr + y * gaussian(np.linspace(imin - 1, imax + 1, discrete_num), mu, sigma)
    return distr
def persistence_vector( dgm, discete_num=500, debug_flag=False,dynamic_range_flag=True):
    ## here filtration only takes sub or super
    result = data_interface(dgm, dynamic_range_flag=dynamic_range_flag)
    data = result['data']
    imax = result['max']
    imin = result['min']
    if debug_flag: print(imax, imin)
    data = rotate_data(data, super_check=False)
    vector = draw_data(data, imax, imin, discrete_num=discete_num)
    vector = np.array(vector).reshape(1, len(vector))
    return vector











def test_viz():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    import numpy

    data = {(0.0, 0.0, 1.0): 0.5874125874125874,
            (0.0, 0.2, 0.8): 0.6593406593406593,
            (0.0, 0.25, 0.75): 0.6783216783216783,
            (0.0, 0.3333, 0.6667): 0.6933066933066933,
            (0.0, 0.4, 0.6): 0.7002997002997003,
            (0.0, 0.4286, 0.5714): 0.7072927072927073,
            (0.0, 0.5, 0.5): 0.7062937062937062,
            (0.0, 0.5714, 0.4286): 0.7042957042957043,
            (0.0, 0.6, 0.4): 0.7022977022977023,
            (0.0, 0.6667, 0.3333): 0.7102897102897103,
            (0.0, 0.75, 0.25): 0.7092907092907093,
            (0.0, 0.8, 0.2): 0.7052947052947053,
            (0.1111, 0.4444, 0.4444): 0.6933066933066933,
            (0.125, 0.375, 0.5): 0.6893106893106893,
            (0.125, 0.5, 0.375): 0.6993006993006993,
            (0.1429, 0.2857, 0.5714): 0.6883116883116883,
            (0.1429, 0.4286, 0.4286): 0.6923076923076923,
            (0.1429, 0.5714, 0.2857): 0.7032967032967034,
            (0.1667, 0.1667, 0.6667): 0.6783216783216783,
            (0.1667, 0.3333, 0.5): 0.6813186813186813,
            (0.1667, 0.5, 0.3333): 0.6993006993006993,
            (0.1667, 0.6667, 0.1667): 0.6823176823176823,
            (0.2, 0.0, 0.8): 0.6833166833166833,
            (0.2, 0.2, 0.6): 0.6783216783216783,
            (0.2, 0.4, 0.4): 0.7012987012987013,
            (0.2, 0.6, 0.2): 0.6833166833166833,
            (0.2, 0.8, 0.0): 0.6693306693306693,
            (0.2222, 0.3333, 0.4444): 0.6923076923076923,
            (0.2222, 0.4444, 0.3333): 0.6973026973026973,
            (0.25, 0.0, 0.75): 0.6853146853146853,
            (0.25, 0.25, 0.5): 0.6863136863136863,
            (0.25, 0.375, 0.375): 0.7002997002997003,
            (0.25, 0.5, 0.25): 0.6973026973026973,
            (0.25, 0.75, 0.0): 0.6663336663336663,
            (0.2727, 0.3636, 0.3636): 0.7012987012987013,
            (0.2857, 0.1429, 0.5714): 0.6793206793206793,
            (0.2857, 0.2857, 0.4286): 0.6943056943056943,
            (0.2857, 0.4286, 0.2857): 0.6943056943056943,
            (0.2857, 0.5714, 0.1429): 0.6763236763236763,
            (0.3, 0.3, 0.4): 0.6993006993006993,
            (0.3, 0.4, 0.3): 0.6993006993006993,
            (0.3333, 0.0, 0.6667): 0.6753246753246753,
            (0.3333, 0.1667, 0.5): 0.6833166833166833,
            (0.3333, 0.2222, 0.4444): 0.6913086913086913,
            (0.3333, 0.3333, 0.3333): 0.7012987012987013,
            (0.3333, 0.4444, 0.2222): 0.6853146853146853,
            (0.3333, 0.5, 0.1667): 0.6833166833166833,
            (0.3333, 0.6667, 0.0): 0.6703296703296703,
            (0.3636, 0.2727, 0.3636): 0.7012987012987013,
            (0.3636, 0.3636, 0.2727): 0.6903096903096904,
            (0.375, 0.125, 0.5): 0.6843156843156843,
            (0.375, 0.25, 0.375): 0.7022977022977023,
            (0.375, 0.375, 0.25): 0.6893106893106893,
            (0.375, 0.5, 0.125): 0.6803196803196803,
            (0.4, 0.0, 0.6): 0.6773226773226774,
            (0.4, 0.2, 0.4): 0.6983016983016983,
            (0.4, 0.3, 0.3): 0.6933066933066933,
            (0.4, 0.4, 0.2): 0.6873126873126874,
            (0.4, 0.6, 0.0): 0.6913086913086913,
            (0.4286, 0.0, 0.5714): 0.6803196803196803,
            (0.4286, 0.1429, 0.4286): 0.6893106893106893,
            (0.4286, 0.2857, 0.2857): 0.6993006993006993,
            (0.4286, 0.4286, 0.1429): 0.6823176823176823,
            (0.4286, 0.5714, 0.0): 0.6933066933066933,
            (0.4444, 0.1111, 0.4444): 0.6903096903096904,
            (0.4444, 0.2222, 0.3333): 0.6983016983016983,
            (0.4444, 0.3333, 0.2222): 0.6873126873126874,
            (0.4444, 0.4444, 0.1111): 0.6853146853146853,
            (0.5, 0.0, 0.5): 0.6893106893106893,
            (0.5, 0.125, 0.375): 0.7022977022977023,
            (0.5, 0.1667, 0.3333): 0.7012987012987013,
            (0.5, 0.25, 0.25): 0.6933066933066933,
            (0.5, 0.3333, 0.1667): 0.6863136863136863,
            (0.5, 0.375, 0.125): 0.6883116883116883,
            (0.5, 0.5, 0.0): 0.7012987012987013,
            (0.5714, 0.0, 0.4286): 0.6913086913086913,
            (0.5714, 0.1429, 0.2857): 0.6943056943056943,
            (0.5714, 0.2857, 0.1429): 0.6873126873126874,
            (0.5714, 0.4286, 0.0): 0.6933066933066933,
            (0.6, 0.0, 0.4): 0.6983016983016983,
            (0.6, 0.2, 0.2): 0.6853146853146853,
            (0.6, 0.4, 0.0): 0.7022977022977023,
            (0.6667, 0.0, 0.3333): 0.7042957042957043,
            (0.6667, 0.1667, 0.1667): 0.6863136863136863,
            (0.6667, 0.3333, 0.0): 0.7042957042957043,
            (0.75, 0.0, 0.25): 0.7042957042957043,
            (0.75, 0.25, 0.0): 0.7092907092907093,
            (0.8, 0.0, 0.2): 0.7102897102897103,
            (0.8, 0.2, 0.0): 0.7122877122877123,
            (1.0, 0.0, 0.0): 0.7102897102897103}

    DATA = numpy.random.rand(20, 3)
    Xs = np.array([i[0] for i in data.keys()])  # DATA[:,0]
    Ys = np.array([i[1] for i in data.keys()])  # DATA[:,0]
    ZZ = np.array([i[2] for i in data.keys()])  # DATA[:,0]
    Zs = np.array(data.values())
    for i in range(len(Xs)):
        assert data[(Xs[i], Ys[i], ZZ[i])] == Zs[i]

    # ======
    ## plot:

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    # 2
    ax = fig.add_subplot(222, projection='3d')
    ax.plot_trisurf(Xs, Ys, Zs, linewidth=0.2, antialiased=True)
    # 3
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(Xs, Ys, Zs)
    # Axes3D.scatter

    fig.tight_layout()

    plt.show()  # or:
    # fig.savefig('3D.png')
def show_plot():
    import numpy as np
    import matplotlib
    matplotlib.get_backend()
    # matplotlib.use('GTK')
    # matplotlib.use('TkAgg')
    # matplotlib.use('GTK3Cairo')
    # matplotlib.use('GTKAgg')
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt

    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()
    plt.close()
def computePD(i, key='null', simplify_flag=0):
    # i = 1; key= 'cc'
    id = i
    io_ = io()
    files = get_swc_files()
    df = get_df(files, id)
    tree = convert2nx(df)
    if simplify_flag == 1:
        tree = simplify_tree(tree)
    g = function_basis(tree, key, simplify_flag)

    dgm = get_diagram(g, key=key)
    export_dgm(id, dgm, files, key)
    print('.'),
    #print('Finish tree %s' % i)
def check_functionval(g, Neuron_dgm):
    vals = dgm_distinct(Neuron_dgm)
    for val in vals:
        find_node_val(g, 'direct_distance', val)
def lifetime_stat(dgm, Y, i):
    # dgm = dgms[110]
    colormap_ = zip(range(1,5),('blue', 'red', 'yellow','green'))
    lifetime = [abs(p.death - p.birth) for p in dgm]
    assert len(lifetime) == len(dgm)
    count_data = np.histogram(lifetime, bins=20)[0]
    count_data = count_data / float(np.sum(count_data))
    import matplotlib.pyplot as plt
    plt.plot(count_data, color = colormap_[Y[i]-1][1], linewidth=1, alpha=0.4)
    plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/viz_tree/new_tree_lifetime/' + 'test' + '.png')
    # plt.close()
    print ('Finish %s'%i)

