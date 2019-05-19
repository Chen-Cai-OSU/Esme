import h5py
import numpy as np
import sys
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/')
from condition import *
import argparse
# parser = argparse.ArgumentParser(discription='generate provider')
# args = parser.parse_args()
# args_dict = vars(args)
# GRAPH = args_dict['graph']
# FUNCTION_TYPE = args_dict['function_type']

def read_data(id=3, type="ricci", hom=0, filtration='sub'):
    """
    :param id:
    :param type:
    :return: a list of tuples
    """
    if hom==1:
        homology = '-1-1st_'
    else:
        homology = '-1-0th_'
    file_directory =DIRECTORY
    file_directory = file_directory + type + "/" + str(id) + homology + filtration+ "PD" + ".txt"
    print(file_directory)

    file = open(file_directory,"r")
    data = file.read();
    file.close()
    data = data.split();
    # print(data)
    # for x in data:
    #     if type(float(x))=='float':
    #         raise ValueError('A very specific bad thing happened.')
    data = [float(x) for x in data]
    imax = 0; imin = 0
    try:
        imax = np.max(data)
        imin = np.min(data)
    except ValueError:
        return {'data':np.array([[0, 1]]), 'max':1,'min':0}
        pass
    # imax = 85
    # imin = -85
    def convert2tuple(data):
        length = len(data)
        assert (length % 2==0)
        tuple = []
        for i in range(0,length,2):
            tuple.append((data[i], data[i+1]))
        return tuple

    if imax!=imin:
        data = [(x-imin)/(imax-imin) for x in data]
    else:
        data = [0,1]
    imax = np.max(data); imin = np.min(data)
    data = convert2tuple(data)
    return {'data':data, 'max':imax,'min':imin}

def read_data_ss(id=3, type='ricci',hom=0):
    # return a numpy array
    def rotate(x, y):
        return np.sqrt(2) / 2 * np.array([x + y, -x + y])

    l1 = np.array(read_data(id, type, hom, 'sub')['data'] + read_data(id, type, hom, 'super')['data'])
    d = np.array(read_data(id, type, hom, 'sub')['data'] + [])
    swap = np.array(read_data(id, type, hom, 'super')['data'] + [(0.0,0.0)])
    assert ((np.shape(swap))[1]==2)
    swap = swap[:,[1,0]]
    return d
read_data_ss()

read_data(3, 'ricci', 1, 'sub')['data']

def read_essential(id=3, type='ricci', hom='0'):
    tmp =  read_data(id, type, int(hom), 'sub')
    data = np.array(tmp['data'])
    # rotate data clockwise by 45 deg. Important.
    # data_rotate = np.zeros(np.shape(data))
    # data_rotate[:,0] = (data[:,0] + data[:,1]) * np.sqrt(2)/2.0
    # data_rotate[:,1] = (-data[:,0] + data[:,1]) * np.sqrt(2)/2.0
    # data = data_rotate
    # print(data)
    max = np.max(data)
    if max!=1:
        print('the max of graph %s in hom %s is %s'%(id, hom, max))
    # assert max==1
    # data_normalized = (data - min)/(max - min)
    n = np.shape(data)[0]
    ess = np.empty([1,2])
    non_ess = np.empty([1,2])
    if hom=='0':
        for i in range(n):
            if ((data[i,1]) < max and (data[i,1]-data[i,0] > 0.01)):
                non_ess = np.append(non_ess, [[data[i,0], data[i,1]]], axis=0)
            elif (data[i,1]==max and (data[i,1]-data[i,0] > 0.01)):
                ess = np.append(ess, [[data[i,0], max]], axis=0)
    elif hom=='1':
        for i in range(n):
            non_ess = np.append(non_ess, [[data[i,1], 1]], axis=0)
        ess = non_ess

    # return {'ess': ess[1:,:], 'non_ess': non_ess[1:,:]}
    return {'ess': ess[1:,:], 'non_ess': non_ess[1:,:]}
read_essential(id=3, type='ricci', hom='0')

# read_essential(3, 'ricci','0')['ess']
# read_essential(3, 'ricci','1')['ess']

import sys
sys.path.append('/home/cai.507/Documents/DeepLearning/nips2017/src/sharedCode/')
from provider import Provider
import numpy as np
import h5py

p = Provider()

def generate_dict(homology, filtration_function, ess_check, Y_RANGE):
    def dict_ini(n):
        d = {}
        for i in range(1, n+1):
            lbl =  str(i)
            d[lbl] = {}
        return d
    n = len(np.unique(Y_RANGE))
    dictionary = dict_ini(n)

    for i in range(1, len(Y_RANGE)):
        tmp = read_essential(i, filtration_function, homology)[ess_check]
        dictionary[ str(1+Y_RANGE[i])][str(i)] = tmp

    # for i in range(1, 1001):
    #     dictionary['label1'][str(i)] = read_essential(i, 'deg_norm',homology)[ess_check]
    # for i in range(1001, 2001):
    #     dictionary['label2'][str(i)] = read_essential(i, 'deg_norm', homology)[ess_check]
    # for i in range(2001, 3001):
    #     dictionary['label3'][str(i)] = read_essential(i, 'deg_norm', homology)[ess_check]
    # for i in range(3001, 4001):
    #     dictionary['label4'][str(i)] = read_essential(i, 'deg_norm', homology)[ess_check]
    # for i in range(4001, 4997):
    #     dictionary['label5'][str(i)] = read_essential(i, 'deg_norm', homology)[ess_check]
    # for i in range(1,1000):
    #     dictionary['label5'][str(i)] = np.random.randn(5,2)
    return dictionary

view1 = generate_dict('0', FUNCTION_TYPE, 'non_ess', Y_RANGE)
view2 = generate_dict('0', FUNCTION_TYPE, 'ess', Y_RANGE)
view3 = generate_dict('1', FUNCTION_TYPE, 'non_ess', Y_RANGE)
view4 = generate_dict('1', FUNCTION_TYPE, 'ess', Y_RANGE)


p.add_view('DegreeVertexFiltration_dim_0', view1)
p.add_view('DegreeVertexFiltration_dim_0_essential', view2)
p.add_view('DegreeVertexFiltration_dim_1', view3)
p.add_view('DegreeVertexFiltration_dim_1_essential', view4)
# p.add_view('DegreeVertexFiltraiton_dim_1_essential', view4)
# label_map = {'label1': 1, 'label2': 2, 'label3': 3, 'label4': 4}

p.add_meta_data({'origin': 'this is dummy text.'})
filename = GRAPH + '_' + FUNCTION_TYPE + '.h5'
p.dump_as_h5('/home/cai.507/Documents/DeepLearning/nips2017/data/dgm_provider/' + filename)
# p.dump_as_h5('/tmp/reddit_5K.h5')

'''
print('==============================================')
print('h5 file:')

def compare(GRAPH, FUNCTION_TYPE):
    filename = GRAPH + '_' + FUNCTION_TYPE + '.h5'
    f = h5py.File('/home/cai.507/Documents/DeepLearning/nips2017/data/dgm_provider/' + filename, 'r+')
    dataset = f['data_views/DegreeVertexFiltration_dim_0/1abel1/3'].value
    f.close()
    print('the saved data for 0 hom for graph 3 is the %s'%dataset)
    og_data = read_data(3, FUNCTION_TYPE, 0, 'sub')['data']
    print('the original data is %s '%og_data)

GRAPH = 'reddit_12K'
# compare(GRAPH, FUNCTION_TYPE)
import h5py
filename = '/home/cai.507/Documents/DeepLearning/nips2017/data/dgm_provider/reddit_12K.h5'
f = h5py.File(filename, 'r+')
dataset = f['data_views/DegreeVertexFiltration_dim_0/label1/3'].value
print(dataset)
f.close
og_data = read_data(3, 'ricci', 0, 'sub')['data']


with h5py.File('/home/cai.57/Documents/DeepLearning/nips2017/data/dgm_provider/reddit_5K.h5', 'r') as f:
    f.visit(lambda n: print(n))

print('==============================================')

p = Provider()
p.read_from_h5('/home/cai.57/Documents/DeepLearning/nips2017/data/dgm_provider/reddit_5K.h5')
print('p.data_views:')
print(p.data_views)
print('')
print('p.str_2_int_label_map:')
print(p.str_2_int_label_map)
print('')
print('p.meta_data :')
print(p.meta_data)

'''

import sys


def read_json(topic = '4chan', file = '7eysrg'):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/reddit_scrape/json/' + topic + '/'
    file += '.json'
    import json
    with open(direct + file) as f:
        data = json.load(f)
    data = data['data']
    def filter(dict):
        dict.pop('body', None)
        return dict

    n = len(data)
    for i in range(n):
        data[i] = filter(data[i])

    import networkx as nx
    G = nx.empty_graph()
    G.add_node(0)
    G.node[0]['author'] = 'DummyNode'

    for i in range(n):
        G.add_node(i+1)
        for key, val in data[i].items():
            G.node[i+1][key] = val
    id_dict = nx.get_node_attributes(G, 'id')
    reverse_id_dict = dict([[v, k] for k, v in id_dict.items()])
    parent_dict = nx.get_node_attributes(G, 'parent_id')
    pairing_dict = {}
    for k, v in parent_dict.items():
        v = reverse_id_dict.get(v[3:], 0)
        parent_dict[k] = v
    edges = [(k, v) for k, v in parent_dict.items()]
    G.add_edges_from(edges)
    assert nx.is_tree(G)
    same_author = lambda u, v: (G.node[u]['author']==G.node[v]['author'] and G.node[u]['author']!='DummyNode')
    for i in range(n+1):
        assert 'author' in G.node[i].keys()
    Q = nx.quotient_graph(G, same_author)
    print(nx.info(Q))
    return Q


