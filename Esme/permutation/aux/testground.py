import sys
import networkx as nx
from .helper import timefunction_precise
sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/pythoncode')
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode')
from .cycle_tools import print_dgm, dgm2diag, diag2dgm, make_direct
from classify_neuron import evaluate_clf, searchclf
from .cycle_tools import remove_zero_col
import numpy as np
from joblib import delayed, Parallel
from sklearn.preprocessing import normalize
import dionysus as d
from .cycle_tools import sw_parallel, dgms2swdgm
# from cycle_basis_v2 import evaluate_tda_kernel
from .cycle_tools import persistence_vector, normalize_dgm, flip_dgm
from collections import Counter
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility

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
    Xs = np.array([i[0] for i in list(data.keys())])  # DATA[:,0]
    Ys = np.array([i[1] for i in list(data.keys())])  # DATA[:,0]
    ZZ = np.array([i[2] for i in list(data.keys())])  # DATA[:,0]
    Zs = np.array(list(data.values()))
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

def dgm_distinct(dgm):
    diag = dgm2diag(dgm)
    diag = dgm2diag(dgm)
    distinct_list = [i[0] for i in diag]
    distinct_list += [i[1] for i in diag]
    distinct_list.sort()
    return distinct_list

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
    print(('The existing PD name is %s '%file_directory[-30:]))

    file = open(file_directory,"r")
    data = file.read(); data = data.split('\r\n');
    Data = []
    # this may need to change for different python version
    for i in range(0,len(data)-1):
        data_i = data[i].split(' ')
        data_i_tuple = (float(data_i[0]), float(data_i[1]))
        Data = Data + [data_i_tuple]
    return Data

def get_swc_files(new_neuron_flag=False, sample_flag=False):
    filename1 = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Experiments/data/data_1268/test.out'
    filename2 = '/home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/Experiments/data/data_1268/test.out'
    try:
        file = open(filename1, 'r')
    except:
        file = open(filename2, 'r')

    from subprocess import check_output
    if new_neuron_flag == True:
        filename = ['Large', 'Ivy:neurogliaform', 'Martinotti',  'Nest',  'Neurogliaform',  'Pyramidal']
        file_number = [147, 17, 123, 94, 78, 452]
        if sample_flag ==True: file_number = [147, 17, 123, 94, 78, 100]
        files = []
        for f in filename:
            n = file_number[filename.index(f)]
            for id in range(1, n+1):
                file = check_output('find  /home/cai.507/Documents/DeepLearning/deep-persistence/New_neuron_data/' + f + '/markram/ -name ' + str("\"") + str(id) + '-*.swc' + str("\""), shell=True).split('\n')[0]
                assert file!=''
                files.append(file)
        assert len(files) == sum(file_number)
        return files

    data = file.read();
    data = data.split('\n');
    data = data[:-1]
    assert len(data) == 1268
    return data

    # pass
    # from subprocess import call, check_output
    # try:
    #     files = check_output('find /Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools/Experiments/data/data_1268 -name *.swc', shell=True).split('\n')[:-1]
    # except:
    #     files = check_output('find /home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/Experiments/data/data_1268 -name \'*.swc\'', shell=True).split('\n')[:-1]
    # assert len(files) == 1268
    # return files

def get_swc_file(i):
    files = get_swc_files()
    print(('swc file is %s' %(files[i-1][-30:])))
    return files[i-1]

def test_correspondence():
    for id in [1, 709, 711, 730, 780, 1000, 1100, 1200]:
        get_swc_file(id)
        null = read_neuronPD(id)
        print ('\n')

def read_data(files, i, verbose=1):
        """
        Read data from NeuronTools
        """
        # file_directory = '/Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools_origin/Experiments/data/data_1268/neuron_nmo_principal/hay+markram/CNG version/cell4.CNG.swc'
        file_directory = files[i-1]
        # file_directory = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Test/2000_manual_tree_T1.swc'
        if verbose == 1:
            print(('The %s -th raw data is %s'%(i, file_directory)))
        file = open(file_directory, "r")
        data = file.read();
        data = data.split('\r\n');
        # Data = []
        # this may need to change for different python version
        # for i in range(0, len(data) - 1):
        #     data_i = data[i].split(' ')
        #     data_i_tuple = (float(data_i[0]), float(data_i[1]))
        #     Data = Data + [data_i_tuple]
        return data

def get_df(files, i, new_neuron_flag=False):
    # i = 100
    data = read_data(files, i)
    if new_neuron_flag == False:
        data = data[0]
        data = data.split('\n')
    for j in range(30):
        if data[j].startswith('#'):
            idx = j
    data = data[idx+1:]
    if data[0].startswith(' '):
        for i in range(len(data)):
            data[i] = data[i][1:] # remove the first letter in the string

    assert data[0].startswith('1 ')

    length = len(data)
    import numpy as np
    data_array = np.array([-1] * 7).reshape(1, 7)
    for i in data[0:length - 1]:
        i = i.split(' ')
        if i[-1] == '':
            i = i.remove('')
        ary = np.array([float(s) for s in i]).reshape(1, 7)
        data_array = np.concatenate((data_array, ary), axis=0)
    import pandas as pd
    colnames = ['id', 'structure', 'x', 'y', 'z', 'radius', 'parent']
    df = pd.DataFrame(data_array, columns=colnames)
    return df[1:]

def distance(i,j, treedf):
    # eculidean distance of two nodes
    import numpy as np
    df = treedf
    coord1 = np.array([df['x'][i], df['y'][i], df['z'][i]])
    coord2 = np.array([df['x'][j], df['y'][j], df['z'][j]])
    dist = np.linalg.norm(coord1-coord2)
    return dist

def distance_(i, j, tree):
    # eculidean distance of two nodes
    import numpy as np
    coord1 = np.array(tree.nodes[i]['coordinates'])
    coord2 = np.array(tree.nodes[j]['coordinates'])
    dist = np.linalg.norm(coord1 - coord2)
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

def function_basis(g, key='dist', simplify_flag = 0):
    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    import time
    start = time.time()
    import networkx as nx
    import sys
    import numpy as np
    from .helper import attribute_mean

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
    closeness_centrality = {k: v / min(closeness_centrality.values()) for k, v in closeness_centrality.items()} # no normalization for debug use
    closeness_centrality = {k: 1.0 / v for k, v in closeness_centrality.items()}
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
                                closeness_centrality.items()}  # no normalization for debug use
        closeness_centrality = {k: 1.0 / v for k, v in closeness_centrality.items()}
        for n in g_ricci.nodes():
            g_ricci.node[n]['cc'] = closeness_centrality[n]

        try:
            g_ricci = ricciCurvature(g, alpha=0.5, weight='length')
            assert list(g_ricci.node.keys()) == list(g.nodes())
            ricci_norm = norm(g, 'ricciCurvature')
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] /= ricci_norm
        except:
            print('RicciCurvature Error for graph, set 0 for all nodes')
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] /= ricci_norm

    # print('Graph of size %s, it takes %s'%(len(g), time.time() - start))
    return g_ricci

def print_node_vals(g, key):
    n = len(g)
    for i in range(1, n+1):
        print((i, g.nodes[i][key]))

def find_node_val(g, key, val):
    import numpy as np
    n = len(g); flag = 0
    for i in range(1, n + 1):
        if np.abs(g.nodes[i][key] - val) < 0.01:
            print((i, g.nodes[i][key]));
            flag = 1
    if flag == 0:
        print('Did not match')

def get_diagram(g, key='dist', typ='tuple', subflag = 'False', int_flag=False):
    # only return 0-homology of sublevel filtration
    # type can be tuple or pd. tuple can be parallized, pd cannot.
    import dionysus as d
    def get_simplices(gi, key='dist'):
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        import networkx as nx
        assert len(list(gi.node)) > 0
        # print(key, gi.node[list(gi.nodes)[2]].keys())
        assert key in list(gi.node[list(gi.nodes)[2]].keys())

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
                print((i, pt.birth, pt.death))

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

def number_lines(i, files):
    file_directory = files[i - 1]
    count = len(open(file_directory).readlines())
    return count

def count_lines():
    for i in range(1, 1000):
        print((i,  number_lines(i)))

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

def computePD(i, key='null', simplify_flag=0):
    # i = 1; key= 'cc'
    id = i
    files = get_swc_files()
    df = get_df(files, id)
    tree = convert2nx(df)
    if simplify_flag == 1:
        tree = simplify_tree(tree)
    g = function_basis(tree, key, simplify_flag)
    import networkx as nx
    # convert2dayu(g)

    dgm = get_diagram(g, key=key)
    export_dgm(id, dgm, files, key)
    print(('.'), end=' ')
    #print('Finish tree %s' % i)

def simplify_tree(tree):
    import networkx as nx
    allnodes = []
    deg_dict = dict(nx.degree(tree))
    for key, val in list(deg_dict.items()):
        if val==2:
            allnodes.append(key)
    for i in allnodes:
        if tree.degree(i) == 2:
            nbr_edges = list(tree.edges(i)) # a list of tuples
            assert len(nbr_edges) == 2
            u = nbr_edges[0][1]
            v = nbr_edges[1][1]
            tree.remove_edges_from(nbr_edges)
            tree.add_edge(u,v, length = distance_(u,v, tree))

    import networkx as nx
    # len(tree)
    # len(tree.edges())
    [len(c) for c in sorted(nx.connected_components(tree), key=len, reverse=True)]
    largest_cc = max(nx.connected_components(tree), key=len)
    subtree = nx.subgraph(tree, largest_cc)
    subtree = nx.convert_node_labels_to_integers(subtree)
    assert nx.is_tree(subtree)
    return subtree

def dgm_uptriangle(dgm):
    for p in dgm:
        assert p.death >= p.birth

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
        print(('dgmx is %s, and dgmy is %s'%(dgmx, dgmy)))
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
    print(('The shape of pairing feature is, ', np.shape(pairing)))
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

@timefunction_precise
def attributes(g, length_flag='discrete', sholl_flag=False):
    import networkx as nx
    import numpy as np
    from collections import Counter

    n = len(g)
    deg_vals = list(dict(nx.degree(g)).values());
    deg_mean_vals = list(nx.get_node_attributes(g, '1_0_deg_mean').values())
    deg_mean_vec = np.histogram(deg_mean_vals, [1+i/10.0 for i in range(40)])[0]
    deg_mean_vec = list(deg_mean_vec)

    deg_std_vals = list(nx.get_node_attributes(g, '1_0_deg_std').values())
    deg_std_vec = np.histogram(deg_std_vals, [i / 20.0 for i in range(40)])[0]
    deg_std_vec = list(deg_std_vec)

    deg_min_vals = list(nx.get_node_attributes(g, '1_0_deg_min').values())
    deg_min_vec = np.histogram(deg_min_vals, list(range(10)))[0]
    deg_min_vec = list(deg_min_vec)

    dist_dev_val = np.array(list(nx.get_node_attributes(g, '1_0_direct_distance_mean').values()))\
                    - np.array(list(nx.get_node_attributes(g, 'direct_distance').values()))
    dist_dev_vec = list(np.histogram(dist_dev_val, list(range(-200, 200, 20)))[0])

    root = list(g.nodes())[0]

    if length_flag == 'discrete':
        dist_vals = list(dict(nx.shortest_path_length(g, root)).values())
        his = Counter(dist_vals)
        assert (max(his.keys()) - min(his.keys()) + 1) == len(list(his.keys()))
        dist_vals = list(his.values())
        assert sum(dist_vals) == len(g)

    elif length_flag == 'continuous':
        try:
            dist_vals = list(dict(nx.shortest_path_length(g, 1, weight='length')).values())
        except:
            print('continuous attribute accidient')
            dist_vals = list(dict(nx.shortest_path_length(g, root, weight='length')).values()) # the initial implementation use node 1 and it seems that 1 is better than root
        his = np.histogram(dist_vals, bins=np.array(list(range(0,5000,10)))) # this can be tuned
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

def check_functionval(g, Neuron_dgm):
    vals = dgm_distinct(Neuron_dgm)
    for val in vals:
        find_node_val(g, 'direct_distance', val)

def scatter_comp():
    import matplotlib.pyplot as plt
    import numpy as np
    data = dgm2diag(Neuron_dgm)
    x = np.array([data[i][0] for i in range(len(data)-1)])
    y = np.array([data[i][1] for i in range(len(data)-1)])
    plt.scatter(x, y, c='b')
    plt.show()

def get_tree(files, i, simplify_flag=1, new_neuron_flag = False):
    # i = 684; new_neuron_flag = True
    id = i
    df = get_df(files, id, new_neuron_flag=new_neuron_flag)

    tree = convert2nx(df)
    if simplify_flag ==1:
        tree = simplify_tree(tree)
    return tree

def getX(i, length_flag='discrete', sholl_flag=False, split_tree_flag=False):
    assert 'trees' in list(globals().keys())
    if split_tree_flag==True:
        # tree_dict = split_trees[i]
        # i = 419
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
            if structure_id not in list(feat.keys()):
                feat[structure_id] = np.zeros((501,))

        feat_ = np.concatenate((feat['1'], feat['2'], feat['3'], feat['4']))
        print((i, np.shape(feat_)))
        return feat_

    tree = trees[i]
    g = function_basis(tree)
    if length_flag=='both':
        discrete = attributes(g, length_flag='discrete')
        continuous = attributes(g, length_flag='continuous')
        return np.concatenate((discrete, continuous), axis=0)
    return attributes(g, length_flag=length_flag, sholl_flag=sholl_flag)

def get_trees(files, n, simplify_flag=1, new_neuron_flag=False):
    # trees = Parallel(n_jobs=1)(delayed(get_tree)(files, i, simplify_flag=simplify_flag, new_neuron_flag=new_neuron_flag) for i in range(1, n + 1))
    trees = []
    for i in range(1, n+1):
        tree = get_tree(files, i, simplify_flag = simplify_flag, new_neuron_flag = new_neuron_flag)
        trees.append(tree)
    return trees

def pd_vector(i, discrete_num = 500, key='direct_distance', dgm_flag=False, unnormalize_flag = False):
    if i % 50 == 0:
        print(('#'), end=' ')
    assert 'trees' in list(globals().keys())
    tree = trees[i]
    g = function_basis(tree, key=key)
    dgm = get_diagram(g, key=key) # not sure here
    if dgm[0].birth > dgm[0].death:
        dgm = flip_dgm(dgm)
    if unnormalize_flag == True:
        return dgm
    dgm = normalize_dgm(dgm)
    if dgm_flag == True:
        return dgm
    feature = persistence_vector(dgm, discete_num=discrete_num)
    print(('.'), end=' ')
    return feature

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
                    print(('Length of nodes %s is not 2'%i), end=' ')
                    print (othernodes)
            # print(i, othernodes)
            angle_ = angle(g, i, othernodes[0], othernodes[1])
            return angle_

def get_dict_key(val_dict, value):
    FirstIndex = lambda a, val, tol: next(i for i, _ in enumerate(a) if np.isclose(_, val, tol))
    a = list(val_dict.values())
    idx = FirstIndex(a, value, tol=1e-6)
    start_key = min(val_dict.keys())
    if abs(val_dict[idx + start_key] - value) > 0.01:
        print(('Looing for %s, idx is %s, found %s' % (value, idx, val_dict[idx])))
    return list(val_dict.keys())[idx]

def reshapeX(X_):
    if np.shape(X_)[0] != 911:
        X = np.vstack(X_)
    if n == 1268:
        try:
            X = np.concatenate((X_[1 - 1:711, :], X_[849:1269, :]), axis=0)
        except:
            X = np.concatenate((X_[1 - 1:711], X_[849:1269]), axis=0)
    return X

def homfeature(i, dgm, key = 'dist'):
    # output the location of homology feature for tree i, node v
    # i = 10
    # dgm = unnormalized_dgms[i]
    assert 'trees' in list(globals().keys())
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

def stat(lis, high_order=False):
    lis = [a for a in lis if a!=0.00123]
    # list = angledgm
    if high_order == True:
        pass
    return np.array([np.min(lis), np.max(lis), np.median(lis), np.mean(lis), np.std(lis)])

def str_feature(structure_dgm):
    feature = np.zeros(16)
    # str_death = [p[0] for p in structure_dgm]
    # str_birth = [p[1] for p in structure_dgm]
    c = Counter(structure_dgm)
    n = sum(c.values())
    for key, val in list(c.items()):
        idx = int((key[0] - 1) * 4 + (key[1] - 1))
        feature[idx] = val / float(n)
    return feature

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
        print(('*'), end=' ')

    deg_dgm = [(p[0][6], p[1][6]) for p in argument_dgm] # deg
    deg_birth = [p[1] for p in deg_dgm]
    deg_death = [p[0] for p in deg_dgm]
    feature4 = np.concatenate((stat(deg_birth), stat(deg_death)))

    cc_dgm = [(p[0][7], p[1][7]) for p in argument_dgm]  # cc
    cc_birth = [p[1] for p in cc_dgm]
    cc_death = [p[0] for p in cc_dgm]
    feature5 = np.concatenate((stat(cc_birth), stat(cc_death)))

    return np.concatenate((feature1, feature2, feature3, feature4, feature5))

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

def split_tree(tree):
    # split the neuron tree by structure type
    # tree = trees[100]
    structure_dict = nx.get_node_attributes(tree, 'structure')
    # print set(structure_dict.values())

    subtree_idx = [key for key, val in list(structure_dict.items()) if val == 1.0]
    subtree_1 = nx.subgraph(tree, subtree_idx)

    subtree_idx = [key for key, val in list(structure_dict.items()) if val == 2.0]
    subtree_2 = nx.subgraph(tree, subtree_idx)

    subtree_idx = [key for key, val in list(structure_dict.items()) if val == 3.0]
    subtree_3 = nx.subgraph(tree, subtree_idx)

    subtree_idx = [key for key, val in list(structure_dict.items()) if val == 4.0]
    subtree_4 = nx.subgraph(tree, subtree_idx)

    return {'1': subtree_1, '2': subtree_2, '3': subtree_3, '4': subtree_4}

def viz_tree(i, show_flag=False):

        assert 'trees' in list(globals().keys())
        assert 'Y' in list(globals().keys())
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
        nx.draw_networkx(tree, pos=pos, node_size=5, with_labels=False, node_color = list(nx.get_node_attributes(tree, 'structure').values()))
        import matplotlib.pyplot as plt
        plt.title('Label:%s'%Y[i])
        plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/viz_tree/new_tree/' + str(i) + '.png')
        if show_flag:
            plt.show()
        plt.close()
        if i % 50 == 0:
            print(('#'), end=' ')

def ExtraFeatBL(i, type='1_2'):
    # get a baseline representation
    # of structure, radius feature
    assert 'trees' in globals()
    tree = trees[i]
    radius_val = list(nx.get_node_attributes(tree, 'radius').values())
    feat1 = stat(radius_val)
    feat2 = np.array([0.0,0.0,0.0,0.0])
    struct_counter = Counter(list(nx.get_node_attributes(tree, 'structure').values()))
    for i in range(4):
        print(struct_counter[float(i) + 1]/float(len(tree)))
        feat2[i] = struct_counter[float(i)+1]/float(len(tree))
    if type == '1_2':
        return np.concatenate((feat1, feat2))
    elif type == '1':
        return feat1
    elif type == '2':
        return feat2
    else:
        raise Exception

def reg_process(X):
    # reshape, remove zero column, and normalize
    X = reshapeX(X)
    X = remove_zero_col(X)
    X = normalize(X, norm='l2', axis=1, copy=True)
    return X

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
    print(('The result is of shape', np.shape(result)))
    return result

def test_feat(X, Y, linear_flag=False, param_print=False, weight_flag=False):
    if weight_flag == False:
        param = searchclf(X, Y, 1002, test_size=t, linear_flag=linear_flag, weight_flag=False)
    else:
        (param, weight) = searchclf(X, Y, 1002, test_size=t, linear_flag=linear_flag, weight_flag=True)
    evaluate_clf(X, Y, param, n_splits=10)
    if param_print:
        print(param)

def lifetime_stat(dgm, Y, i):
    # dgm = dgms[110]
    colormap_ = list(zip(list(range(1,5)),('blue', 'red', 'yellow','green')))
    lifetime = [abs(p.death - p.birth) for p in dgm]
    assert len(lifetime) == len(dgm)
    count_data = np.histogram(lifetime, bins=20)[0]
    count_data = count_data / float(np.sum(count_data))
    import matplotlib.pyplot as plt
    plt.plot(count_data, color = colormap_[Y[i]-1][1], linewidth=1, alpha=0.4)
    plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/viz_tree/new_tree_lifetime/' + 'test' + '.png')
    # plt.close()
    print(('Finish %s'%i))

# for i in range(1,400,10):
#     lifetime_stat(dgms[i], Y, i)
# plt.close()

if __name__ == '__main__':
    new_neuron_flag = True; sample_flag=True; sholl_flag=True;
    filtration_type = 'direct_distance'; t = 0.1
    (n, Y) = set_label(new_neuron_flag, sample_flag)
    location = '/home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/cachedir/'

    # two ways to get trees; from memeory or compute on the fly.
    files = get_swc_files(new_neuron_flag, sample_flag=sample_flag)
    trees = Parallel(n_jobs=-1)(delayed(get_tree)(files, i, simplify_flag=1, new_neuron_flag=new_neuron_flag) for i in range(1, n + 1))

    # simple stats
    simpleStat_ = Parallel(n_jobs=-1)(delayed(getX)(i, length_flag='continuous', sholl_flag=sholl_flag, split_tree_flag=False) for i in range(n))
    simpleStat = reg_process(simpleStat_)

    # struct, radius stats
    extraBLfeat_ = Parallel(n_jobs=-1)(delayed(ExtraFeatBL)(i, type = '1_2') for i in range(n))
    extraBLfeat = reg_process(extraBLfeat_)

    X = np.concatenate((simpleStat, extraBLfeat), axis=1); X = reg_process(X) # seems that reg_process one more time helps
    test_feat(X, Y)

    # PD vector
    pdX = Parallel(n_jobs=-1)(delayed(pd_vector)(i, discrete_num=100, key=filtration_type) for i in range(n))
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
    print(sample_mat)
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
            print(tda_kernel_data_)

    # Combine both
    X1 = np.vstack(X_); X1 = remove_zero_col(X1); X1 = normalize(X1, norm='l2', axis=1, copy=True)
    X2 = pdX; X2 = remove_zero_col(X2); X2 = normalize(X2, norm='l2', axis=1, copy=True)
    X = np.concatenate((X1, X2), axis=1)
    if n == 1268:
        X = np.concatenate((X[1 - 1:711, :], X[849:1269, :]), axis=0)
    print(('Shape of X is', np.shape(X)))
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
        from .cycle_tools import persistence_vector
        feature = persistence_vector(dgm)
        np.shape(feature)


    tuned_parameters=  [{'kernel': ['linear'], 'C': [ 0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 200, 500, 1000]},
                        {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10], 'C': [0.1, 1, 10, 100, 1000]}]
    searchclf(1001)

    from joblib import delayed, Parallel
    Parallel(n_jobs=-1)(delayed(computePD)(i, 'ricciCurvature', 1) for i in range(1, 1269))

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
