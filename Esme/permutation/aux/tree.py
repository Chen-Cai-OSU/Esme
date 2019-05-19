import argparse
import sys
import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from sklearn.preprocessing import normalize
from collections import Counter
import pandas as pd
from subprocess import check_output

sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/pythoncode')
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode')
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/')

from aux.helper import timefunction_precise
from aux.tools import print_dgm, dgm2diag, diag2dgm, make_direct, remove_zero_col, normalize_dgm, flip_dgm, assert_dgm_above, assert_dgm_below
from aux.sw import sw_parallel, dgms2swdgm
from aux.tree_util import function_basis, convert2nx, distance, set_label, reg_process, test_feat, reshapeX
from aux.tree_util import aggregation_feature, pairing_feature, dgm_decoration, searchclf, evaluate_clf, stat
from aux.tree_util import find_node_val, distance_, split_tree, persistence_vector
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility

def read_neuronPD(id, new_neuron_flag = False):
    """
    Read data from NeuronTools
    :return: a list of tuples
    """
    from subprocess import check_output
    try:
        file_directory = check_output('find /Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools/persistence_diagrams/3_data_1268 -depth 1 -name ' +  str(id) + '_*', shell=True).split('\n')[:-1][0]
    except:
        file_directory = check_output('find /home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/persistence_diagrams/3_data_1268 -maxdepth 1 -name ' + str('\'') + str(id) + '_*' + str('\''), shell=True).split('\n')[:-1][0]
    print('The existing PD name is %s '%file_directory[-30:])

    file = open(file_directory,"r")
    data = file.read(); data = data.split('\r\n');
    Data = []
    # this may need to change for different python version
    for i in range(0,len(data)-1):
        data_i = data[i].split(' ')
        data_i_tuple = (float(data_i[0]), float(data_i[1]))
        Data = Data + [data_i_tuple]
    return Data


class Io():
    def __init__(self, new_neuron_flag=False, sample_flag=False):
        self.new_neuron_flag = new_neuron_flag
        self.sample_flag = sample_flag
        self.files = []

    def get_swc_files(self):
        filename1 = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Experiments/data/data_1268/test.out'
        filename2 = '/home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/Experiments/data/data_1268/test.out'
        try:
            file = open(filename1, 'r')
        except IOError:
            file = open(filename2, 'r')

        if self.new_neuron_flag == True:
            filename = ['Large', 'Ivy:neurogliaform', 'Martinotti',  'Nest',  'Neurogliaform',  'Pyramidal']
            file_number = [147, 17, 123, 94, 78, 452]
            if self.sample_flag ==True: file_number = [147, 17, 123, 94, 78, 100]
            files = []
            for f in filename:
                n = file_number[filename.index(f)]
                for id in range(1, n+1):
                    file = check_output('find  /home/cai.507/Documents/DeepLearning/deep-persistence/New_neuron_data/' + f + '/markram/ -name ' + str("\"") + str(id) + '-*.swc' + str("\""), shell=True).split('\n')[0]
                    assert file!=''
                    files.append(file)
            assert len(files) == sum(file_number)
            return files
        else:
            data = file.read()
            data = data.split('\n')
            data = data[:-1]
            assert len(data) == 1268
            return data

    def get_swc_file(self, i):
        files = self.get_swc_files()
        print ('swc file is %s' % (files[i - 1][-30:]))
        return files[i - 1]

def get_diagram(g, key='dist', typ='tuple', subflag = 'False', int_flag=False):
    # only return 0-homology of sublevel filtration
    # type can be tuple or pd. tuple can be parallized, pd cannot.
    import dionysus as d
    def get_simplices(gi, key='dist'):
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        import networkx as nx
        assert len(list(gi.node)) > 0
        # print(key, gi.node[list(gi.nodes)[2]].keys())
        assert key in gi.node[list(gi.nodes)[2]].keys()

        simplices = list()
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2]['length']):
            # tup = ([u, v], data['length'])
            tup = ([u, v], min(gi.nodes[u][key], gi.nodes[v][key]))
            simplices.append(tup)

        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            tup = ([v], data[key])
            simplices.append(tup)

        return simplices

    simplices = get_simplices(g, key=key)

    def compute_PD(simplices, sub=True, inf_flag='False'):
        import dionysus as d
        f = d.Filtration()
        for simplex, time in simplices:
            f.append(d.Simplex(simplex, time))
        if sub == True:
            f.sort()
        elif sub == False:
            f.sort(reverse=True)
        for s in f:
            continue
            print(s)
        m = d.homology_persistence(f)
        # for i,c in enumerate(m):
        #     print(i,c)
        dgms = d.init_diagrams(m, f)
        # print(dgms)
        for i, dgm in enumerate(dgms):
            for pt in dgm:
                continue
                print(i, pt.birth, pt.death)

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
                diagram = d.Diagram(pt_list)
                dgms_list[i] = diagram

            return dgms_list

        if inf_flag == 'False':
            dgms = del_inf(dgms)

        return dgms

    super_dgms = compute_PD(simplices, sub=False)
    sub_dgms = compute_PD(simplices, sub=True)
    n_node = len(g.nodes)
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
        if int_flag == True:
            return int_dgm(sub_dgms[0])
        else:
            return sub_dgms[0]
    elif subflag=='False':
        if int_flag == True:
            return int_dgm(super_dgms[0])
        else:
            return super_dgms[0]

@timefunction_precise
def attributes(g, length_flag='discrete', sholl_flag=False):
    n = len(g)
    deg_vals = dict(nx.degree(g)).values()
    deg_mean_vals = nx.get_node_attributes(g, '1_0_deg_mean').values()
    deg_mean_vec = np.histogram(deg_mean_vals, [1+i/10.0 for i in range(40)])[0]
    deg_mean_vec = list(deg_mean_vec)

    deg_std_vals = nx.get_node_attributes(g, '1_0_deg_std').values()
    deg_std_vec = np.histogram(deg_std_vals, [i / 20.0 for i in range(40)])[0]
    deg_std_vec = list(deg_std_vec)

    deg_min_vals = nx.get_node_attributes(g, '1_0_deg_min').values()
    deg_min_vec = np.histogram(deg_min_vals, range(10))[0]
    deg_min_vec = list(deg_min_vec)

    dist_dev_val = np.array(nx.get_node_attributes(g, '1_0_direct_distance_mean').values())\
                    - np.array(nx.get_node_attributes(g, 'direct_distance').values())
    dist_dev_vec = list(np.histogram(dist_dev_val, range(-200, 200, 20))[0])

    root = list(g.nodes())[0]

    if length_flag == 'discrete':
        dist_vals = dict(nx.shortest_path_length(g, root)).values()
        his = Counter(dist_vals)
        assert (max(his.keys()) - min(his.keys()) + 1) == len(his.keys())
        dist_vals = his.values()
        assert sum(dist_vals) == len(g)

    elif length_flag == 'continuous':
        try:
            dist_vals = dict(nx.shortest_path_length(g, 1, weight='length')).values()
        except:
            print('continuous attribute accidient')
            dist_vals = dict(nx.shortest_path_length(g, root, weight='length')).values() # the initial implementation use node 1 and it seems that 1 is better than root
        his = np.histogram(dist_vals, bins=np.array(range(0,5000,10))) # this can be tuned
        his = his[0]
        assert np.sum(his)==len(dist_vals)
        dist_vals = his

    max_distance = max(dist_vals)
    ave_dist = np.mean(dist_vals)
    std_dist = np.std(dist_vals)

    max_deg = max(deg_vals);
    ave_deg = np.mean(deg_vals)
    std_deg = np.std(deg_vals)

    deg_vec = [0]*100; dist_vec = [0] * 500

    for i in range(len(dist_vals)):
        dist_vec[i] = dist_vals[i]

    for i in range(len(deg_vec)):
        deg_vec[i] = deg_vals.count(i+1)

    assert sum(deg_vec) == sum(dist_vals) == len(g.nodes())
    if sholl_flag==False:
        attributes = [n, max_distance, ave_dist, std_dist, max_deg, ave_deg, std_deg] + deg_vec + dist_vec + deg_mean_vec + deg_std_vec + deg_min_vec # + dist_dev_vec
    elif sholl_flag==True:
        attributes = [n] + dist_vec

    return np.array(attributes)

def ExtraFeatBL(i, type='1_2'):
    # get a baseline representation
    # of structure, radius feature
    assert 'trees' in globals()
    tree = trees[i]
    radius_val = nx.get_node_attributes(tree, 'radius').values()
    feat1 = stat(radius_val)
    feat2 = np.array([0.0,0.0,0.0,0.0])
    struct_counter = Counter(nx.get_node_attributes(tree, 'structure').values())
    for i in range(4):
        print struct_counter[float(i) + 1]/float(len(tree))
        feat2[i] = struct_counter[float(i)+1]/float(len(tree))
    if type == '1_2':
        return np.concatenate((feat1, feat2))
    elif type == '1':
        return feat1
    elif type == '2':
        return feat2
    else:
        raise Exception

def unwrap_get_tree(*arg, **kwarg):
    # http://qingkaikong.blogspot.com/2016/12/python-parallel-method-in-class.html
    return Trees.get_tree(*arg, **kwarg)

class Trees():
    def __init__(self, files, new_neuron_flag=False, simplify_flag=1):
        self.files = files
        self.new_neuron_flag = new_neuron_flag
        self.simplify_flag = simplify_flag

    @staticmethod
    def read_data(files, i, verbose=1):
        """
        Read data from NeuronTools
        """
        # file_directory = '/Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools_origin/Experiments/data/data_1268/neuron_nmo_principal/hay+markram/CNG version/cell4.CNG.swc'
        file_directory = files[i - 1]
        # file_directory = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Test/2000_manual_tree_T1.swc'
        if verbose == 1:
            print ('The %s -th raw data is %s' % (i, file_directory))
        file = open(file_directory, "r")
        data = file.read()
        data = data.split('\r\n')
        # Data = []
        # this may need to change for different python version
        # for i in range(0, len(data) - 1):
        #     data_i = data[i].split(' ')
        #     data_i_tuple = (float(data_i[0]), float(data_i[1]))
        #     Data = Data + [data_i_tuple]
        return data

    def get_df(self, i):
        # i = 100
        data = self.read_data(self.files, i)
        if self.new_neuron_flag == False:
            data = data[0]
            data = data.split('\n')
        for j in range(30):
            if data[j].startswith('#'):
                idx = j
        data = data[idx + 1:]
        if data[0].startswith(' '):
            for i in range(len(data)):
                data[i] = data[i][1:]  # remove the first letter in the string
        assert data[0].startswith('1 ')

        length = len(data)
        data_array = np.array([-1] * 7).reshape(1, 7)
        for i in data[0:length - 1]:
            i = i.split(' ')
            if i[-1] == '':
                i = i.remove('')
            ary = np.array([float(s) for s in i]).reshape(1, 7)
            data_array = np.concatenate((data_array, ary), axis=0)
        colnames = ['id', 'structure', 'x', 'y', 'z', 'radius', 'parent']
        df = pd.DataFrame(data_array, columns=colnames)
        return df[1:]

    @staticmethod
    def simplify_tree(tree):
        deg2nodes = []
        deg_dict = dict(nx.degree(tree))
        for key, val in deg_dict.items():
            if val == 2: deg2nodes.append(key)
        for i in deg2nodes:
            if tree.degree(i) == 2:
                nbr_edges = list(tree.edges(i))  # a list of tuples
                assert len(nbr_edges) == 2
                u = nbr_edges[0][1]
                v = nbr_edges[1][1]
                tree.remove_edges_from(nbr_edges)
                tree.add_edge(u, v, length=distance_(u, v, tree))
        [len(c) for c in sorted(nx.connected_components(tree), key=len, reverse=True)]
        largest_cc = max(nx.connected_components(tree), key=len)
        subtree = nx.subgraph(tree, largest_cc)
        subtree = nx.convert_node_labels_to_integers(subtree)
        assert nx.is_tree(subtree)
        return subtree

    def get_tree(self, i):
        # i = 684; new_neuron_flag = True
        df = self.get_df(i)
        tree = convert2nx(df)
        if self.simplify_flag ==1:
            tree = self.simplify_tree(tree)
        return tree

    def get_trees(self, n):
        trees = Parallel(n_jobs=-1)\
            (delayed(unwrap_get_tree)(self, i) for i in range(1, n + 1))
        return trees
        # serial version
        # trees = []
        # for i in range(1, n + 1):
        #     tree = unwrap_get_tree(self, i, simplify_flag=simplify_flag)
        #     trees.append(tree)
        # return trees

def getX(i, length_flag='discrete', sholl_flag=False, split_tree_flag=False):
    assert 'trees' in globals().keys()
    if split_tree_flag==True:
        tree_dict = split_tree(trees[i])
        feat = {}
        for structure_id in ['1', '2', '3', '4']:
            tree = tree_dict[structure_id]
            if len(tree) < 5:
                feat[structure_id] = np.zeros((501,))
                continue
            # get the component
            subtrees = [subtree for subtree in sorted(nx.connected_component_subgraphs(tree), key=len, reverse=True)]
            for i_ in range(len(subtrees)):
                g = subtrees[i_]
                if len(g) < 5: continue
                if length_flag == 'both':
                    discrete = attributes(g, length_flag='discrete')
                    continuous = attributes(g, length_flag='continuous')
                    if i_ == 0:
                        feat[structure_id] =  np.concatenate((discrete, continuous), axis=0)
                    else:
                        feat[structure_id] += np.concatenate((discrete, continuous), axis=0)
                else:
                    if i_ == 0:
                        feat[structure_id] =  attributes(g, length_flag=length_flag, sholl_flag=sholl_flag)
                    else:
                        feat[structure_id] += attributes(g, length_flag=length_flag, sholl_flag=sholl_flag)

            # one more check
            if structure_id not in feat.keys():
                feat[structure_id] = np.zeros((501,))
        feat_ = np.concatenate((feat['1'], feat['2'], feat['3'], feat['4']))
        print(i, np.shape(feat_))
        return feat_

    g = function_basis(trees[i])
    if length_flag=='both':
        discrete = attributes(g, length_flag='discrete')
        continuous = attributes(g, length_flag='continuous')
        return np.concatenate((discrete, continuous), axis=0)
    else:
        return attributes(g, length_flag=length_flag, sholl_flag=sholl_flag)

def assert_dgm_below(dgm):
    i = 0
    for p in dgm:
        if p.birth < p.death:
            raise Exception('Dgm not above diagonal. Found %s point, birth is %s, and death is %s'%(i, p.birth,p.death))
        i = i + 1

def pd_vector(i, discrete_num = 500, key='direct_distance', dgm_flag=False, unnormalize_flag = False):
    assert 'trees' in globals().keys()
    tree = trees[i]
    g = function_basis(tree, key=key)
    dgm = get_diagram(g, key=key) # not sure here
    if dgm[0].birth > dgm[0].death:
        dgm = flip_dgm(dgm)
    if unnormalize_flag == True: return dgm
    dgm = normalize_dgm(dgm)
    if dgm_flag == True:
        if i % 50 == 0: print ('#'),
        return dgm
    assert_dgm_above(dgm)
    feature = persistence_vector(dgm, discete_num=discrete_num)
    if i % 50 == 0: print ('#'),
    return feature

def extra_features_processor(extra_features_, type = '1_2_3'):
    # seperate structure, radius, and angle
    np.shape(extra_features_)[1] == 31 + 10 + 10
    feat_null = np.zeros((np.shape(extra_features_)[0],0))
    feat1 = extra_features_[:,0:16] if '1' in type else feat_null
    feat2 = extra_features_[:, 16:26]if '2' in type else feat_null
    feat3 = extra_features_[:, 26:31]if '3' in type else feat_null
    feat4 = extra_features_[:, 31:41] if '4' in type else feat_null
    feat5 = extra_features_[:, 41:51] if '5' in type else feat_null
    result =  np.concatenate((feat1, feat2, feat3, feat4, feat5), axis=1)
    print('The result is of shape', np.shape(result))
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--graph', default='mutag', help="mutag, ptc, reddit...")
parser.add_argument('filtration', default='direct_distance')
parser.add_argument('new_neuron_flag')

if __name__ == '__main__':
    new_neuron_flag = True
    sample_flag=True
    sholl_flag=True
    filtration_type = 'direct_distance'
    (n, Y) = set_label(new_neuron_flag, sample_flag)

    # two ways to get trees; from memeory or compute on the fly.
    io =  Io(new_neuron_flag=new_neuron_flag, sample_flag=sample_flag) #TODO better name
    files = io.get_swc_files()
    t = Trees(files, new_neuron_flag=new_neuron_flag)
    trees = t.get_trees(n)

    # simple stats
    simpleStat_ = Parallel(n_jobs=-1)(delayed(getX)(i, length_flag='continuous', sholl_flag=sholl_flag, split_tree_flag=False) for i in range(n))
    simpleStat = reg_process(simpleStat_, n) #TODO comment back

    # struct, radius stats
    extraBLfeat_ = Parallel(n_jobs=-1)(delayed(ExtraFeatBL)(i, type = '1_2') for i in range(n))
    extraBLfeat = reg_process(extraBLfeat_, n)# TODO comment back

    X = np.concatenate((simpleStat, extraBLfeat), axis=1);
    X = reg_process(X, n) # seems that reg_process one more time helps
    test_feat(X, Y, 0.1)

    # PD vector
    pdX = Parallel(n_jobs=1)(delayed(pd_vector)(i, discrete_num=100, key=filtration_type) for i in range(n))
    for i in range(3):
        pd_vector(i, key=filtration_type)


    sys.exit()
    pdX = np.vstack(pdX); X = reshapeX(pdX); X = remove_zero_col(X)
    X = normalize(X, norm='l2', axis=1, copy=True)
    test_feat(X, Y)

    # PD baseline representation
    dgms = Parallel(n_jobs=-1)(delayed(pd_vector)(i, dgm_flag = True, key=filtration_type) for i in range(n))
    dgms_deg = Parallel(n_jobs=-1)(delayed(pd_vector)(i, dgm_flag = True, key='deg') for i in range(n))
    dgms_cc = Parallel(n_jobs=-1)(delayed(pd_vector)(i, dgm_flag=True, key='cc') for i in range(n))

    aggregate_feat = reshapeX(aggregation_feature(dgms))
    (pairing_feat, cor_dict, inv_cor_dict) = pairing_feature(dgms, n_bin=20, cor_flag=True)
    pairing_feat = reshapeX(pairing_feat)

    unnormalized_dgms = Parallel(n_jobs=-1)(delayed(pd_vector)(i, dgm_flag=True, key=filtration_type, unnormalize_flag=True) for i in range(n))
    extra_features_ = Parallel(n_jobs=-1)(delayed(dgm_decoration)(i) for i in range(n)) # computational expansive;
    extra_features = np.array(np.vstack(extra_features_))
    extra_features = extra_features_processor(extra_features, type='1_2_3')
    extra_features__ = reg_process(extra_features)

    X = np.concatenate((aggregate_feat, extra_features__), axis=1)
    # X = np.concatenate((aggregate_feat, extraBLfeat), axis=1)
    X = pairing_feat
    (param, weight) = searchclf(X, Y, 1002, test_size=t, linear_flag=True, weight_flag=True)
    evaluate_clf(X, Y, param, n_splits=10)

    sys.exit()


    thres = np.mean(abs(weight))
    sig_feat_indicator = (abs(weight) >  thres)
    pos_list = []
    for i in range(np.shape(sig_feat_indicator)[1]):
        if sig_feat_indicator[0,i] == True:
            pos_list.append(i)
    len(pos_list)
    pdpt_pos = [cor_dict[j] for j in pos_list]
    pdpt_pos = [(j / 20, j % 20) for j in pdpt_pos]


    sample_mat = np.zeros((20,20))
    for idx in pdpt_pos:
        sample_mat[idx[0], 19 - idx[1]] = weight[0, inv_cor_dict[idx[0]*20 + idx[1]]]
    print sample_mat
    import matplotlib.pyplot as plt
    im = plt.matshow(sample_mat)

    plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/viz_tree/pd.png')
    plt.show()



    test_feat(X,Y, linear_flag=True, param_print=True, weight_flag=True)


    # sliced wasserstein
    if n == 1268:
        dgms = np.concatenate((dgms[1 - 1:711], dgms[849:1269]), axis=0)
    for kernel_type in ['sw']:
        tda_kernel_data = (0, 0, {})
        # for bandwidth in [0.01, 0.05]:
        for bandwidth in [10]:
            (tda_kernel, t1) = sw_parallel(dgms2swdgm(dgms), dgms2swdgm(dgms), parallel_flag=True,
                                           kernel_type=kernel_type, n_directions=10, bandwidth=bandwidth)
            tda_kernel_data_ = evaluate_tda_kernel(tda_kernel, Y, [0])
            print tda_kernel_data_

    # Combine both
    X1 = np.vstack(X_); X1 = remove_zero_col(X1); X1 = normalize(X1, norm='l2', axis=1, copy=True)
    X2 = pdX; X2 = remove_zero_col(X2); X2 = normalize(X2, norm='l2', axis=1, copy=True)
    X = np.concatenate((X1, X2), axis=1)
    if n == 1268:
        X = np.concatenate((X[1 - 1:711, :], X[849:1269, :]), axis=0)
    print('Shape of X is', np.shape(X))
    param = searchclf(X, Y, 1002, test_size=t)
    # param = {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
    evaluate_clf(X, Y, param, n_splits=10)

    sys.exit()

    for i in range(1, 1269):
        i = 923
        id = i
        neuronPD = read_neuronPD(id)
        Neuron_dgm = diag2dgm(neuronPD)
        print_dgm(Neuron_dgm)

        # tree = trees[i]
        tree = get_tree(files, id, simplify_flag=0)
        g = function_basis(tree)
        dgm = get_diagram(g,key='direct_distance')
        print_dgm(dgm)
        feature = persistence_vector(dgm)
        np.shape(feature)


    tuned_parameters=  [{'kernel': ['linear'], 'C': [ 0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 200, 500, 1000]},
                        {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10], 'C': [0.1, 1, 10, 100, 1000]}]
    searchclf(1001)

    from joblib import delayed, Parallel
    Parallel(n_jobs=-1)(delayed(computePD)(i, 'ricciCurvature', 1) for i in range(1, 1269))

