import pickle
import os
import time
import numpy as np
import networkx as nx
import sys


from Esme.helper.io import make_dir
from Esme.helper.time import timefunction
from Esme.helper.format import precision_format

@timefunction
def load_existing_graph(graph, file):
    start = time.time()

    if os.path.isfile(file):
        print('Loading existing files')
        if graph == 'reddit_12K':
            file = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/', 'reddit_12K' + '.graph')
            with open(file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            graphs, labels = data['graph'], data['labels']
        else:
            with open(file, 'rb') as f:
                graphs, labels = pickle.load(f, encoding='latin1')


        print(('Loading takes %s' % precision_format((time.time() - start),1)))
        if graph == 'ptc': graphs[151] = graphs[152] # small hack
        return graphs, labels


def load_graph(graph, debug=False, single_graph_flag=True):
    # exptect label to be numpy.ndarry of shape (n,). However protein_data is different so have to handle it differently
    """
    :param graph: str:
    :param debug:
    :param single_graph_flag:
    :return: graphs and lables
    """

    assert type(graph) == str
    dir = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence', graph, 'LearningFiltration')
    make_dir(dir)
    inputFile = os.path.join(dir, 'graph+label')

    if os.path.isfile(inputFile):
        graphs, labels = load_existing_graph(graph, inputFile)
        return graphs, labels

    print('Start Loading from dataset')
    file = os.path.join("/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset",  graph + ".graph")
    if not os.path.isfile(file):
        file = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets',  graph + '.graph')

    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    graphs = data['graph']
    if graph == 'ptc': graphs[151] = graphs[152]
    labels = data['labels']

    if debug:
        print((graph), end=' ')
        print((type(labels),))
        print((np.shape(labels)))

    if graph == 'protein_data':
        labels = labels.reshape(1113, 1)
        labels = np.array([-1] * 663 + [1] * 450)
    elif graph == ('nci1' or 'nci109'):
        labels = np.sign(labels - 0.5)

    print('Finish Loading graphs')
    outputFile = dir + '/graph+label'
    fw = open(outputFile, 'wb')
    dataset = (graphs, labels)
    pickle.dump(dataset, fw)
    fw.close()
    print('Finish Saving data for future use')
    return graphs, labels


def convert2nx(graph, i, print_flag=False):
    """
    :param graph: python dict
    :param i: index
    :param print_flag: bool
    :return: a networkx graph
    """

    keys = graph.keys()
    try:
        assert list(keys) == list(range(len(graph.keys())))
    except AssertionError:
        print('%s graph has non consecutive keys'%i)
        print('Missing nodes are the follwing:')
        for i in range(max(graph.keys())):
            if i not in graph.keys():
                print(i),

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
    for i in keys:
        if graph[i]['label']=='':
            gi.node[i]['label'] = 1
            # continue
        try:
            gi.node[i]['label'] = graph[i]['label'][0]
        except TypeError: # modifications for reddit_binary
            gi.node[i]['label'] = graph[i]['label']
        except IndexError:
            gi.node[i]['label'] = 0 # modification for imdb_binary

    assert len(gi.node) == len(graph.keys())
    gi.remove_edges_from(gi.selfloop_edges())
    if print_flag:
        print('graph: %s, n_nodes: %s, n_edges: %s' %(i, len(gi), len(gi.edges)) )

    return gi

@timefunction
def load_graphs(dataset = 'mutag', labels_only = False):
    """
    Serial version of loading graphs
    :param dataset:
    :param labels_only: if True, skip converting graph dict to nx graphs. This can save some time.
    :return: a list of nx graphs
    """
    graphs_dict, labels = load_graph(dataset)
    if labels_only:
        return None, labels
    gs = [] # a list of nx graph
    for i in range(len(labels)):
        gi = convert2nx(graphs_dict[i], i)
        gs.append(gi)
    assert len(gs) == len(labels)
    return gs, labels

def component_graphs(g, threshold = 4):
    """
    if a graph is not connected, return a list of subgraphs of size no less than threshold(4)
    :param g:
    :param threshold: if a subgraph is less than threshold, ignore it
    :return:
    """
    # gs, _ = load_graphs('nci1') # a example where input graphs has three components of size (19, 4, 4)
    # g = gs[60]
    if nx.is_connected(g):
        return [g]
    else:
        components = list(nx.connected_component_subgraphs(g))
        assert sum(map(len, components)) == len(g)
        components = [g_ for g_ in components if len(g_) >= threshold]
        components = [nx.convert_node_labels_to_integers(g_) for g_ in components]
        return components

if __name__=='__main__':
    graphs_dict, labels = load_graph('mutag')
    g = convert2nx(graphs_dict[10], 10)
    print(type(g))