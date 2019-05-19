""" collabration network dataset. """
import os
import pickle

import networkx as nx
import numpy as np

from Esme.applications.motif.NRL.src.classification import ArgumentParser, ArgumentDefaultsHelpFormatter
from Esme.dgms.compute import alldgms
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.format import export
from Esme.dgms.kernel import sw_parallel
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.function import fil_strategy
from Esme.ml.svm import classifier


def load_graph(graph, debug='off', single_graph_flag=True):
    # exptect label to be numpy.ndarry of shape (n,). However protein_data is different so have to handle it differently
    assert type(graph) == str
    print('Start Loading from dataset')
    file = os.path.join("/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/", graph + ".graph")
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    graphs, labels = data['graph'], data['labels']
    return graphs, labels

def convert2nx(graph, i, print_flag='False'):
    # graph: python dict
    keys = graph.keys()
    try:
        assert keys == range(len(graph.keys()))
    except AssertionError:
        # print('%s graph has non consecutive keys'%i)
        # print('Missing nodes are the follwing:')
        for i in range(max(graph.keys())):
            if i not in graph.keys(): print(i),

    # add nodes
    gi = nx.Graph()
    for i in keys: gi.add_node(i) # change from 1 to i.
    assert len(gi) == len(keys)

    # add edges
    for i in keys:
        for j in graph[i]['neighbors']:
            if j > i:
                gi.add_edge(i, j)
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
    assert len(gi.node) == len(graph.keys())
    gi.remove_edges_from(gi.selfloop_edges())
    if print_flag=='True': print('graph: %s, n_nodes: %s, n_edges: %s' %(i, len(gi), len(gi.edges)) )
    return gi

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--fil_method", default='edge', type=str, help='Filtration method"')
parser.add_argument("--dataset", default='imdb_binary', type=str, help='dataset')

def is_complete(g):
    if len(g.edges()) == len(g) * (len(g)-1) * 0.5:
        print('It is complete graph')
    else:
        print('It is NOT complete graph')

if __name__ == '__main__':
    # sys.argv = ['graph/2sbm_gc.py']
    args = parser.parse_args()
    fil_method = args.fil_method
    dataset = args.dataset
    zigzag = True if fil_method == 'combined' else False
    graphs, labels = load_graph(dataset, debug='off', single_graph_flag=False)
    gs = []
    nodefeat = 'cc'

    for i in range(len(graphs)):
        g = convert2nx(graphs[i], i)
        g.remove_edges_from(g.selfloop_edges())
        assert list(g.selfloop_edges()) == []
        gs.append(g)
    edge_kwargs = {'h': 0.3, 'edgefunc': 'edge_prob'}

    for i in range(len(gs)):
        if i % 10 ==0: print('.', end='')
        g = gs[i]
        lp = LaplacianEigenmaps(d=1)
        try:
            if nodefeat == 'laplacian':
                lp.learn_embedding(g, weight='weight')
                lapfeat = lp.get_embedding()
            elif nodefeat == 'cc':
                lapfeat = lp.clustering_coefficient(g)
        except:
            lapfeat = np.array([1/np.sqrt(len(g))]*len(g)).reshape((len(g), 1))
            print(is_complete(g))
            print('Error for graph %s'%i)
        gs[i] = fil_strategy(g, lapfeat, method=fil_method, viz_flag=False, **edge_kwargs)

    print('Finish computing lapfeat')
    dgms = alldgms(gs, radius=float('inf'), dataset='', recompute_flag=True, method='serial', n=len(gs), zigzag=zigzag)  # compute dgms in parallel
    # sys.exit()

    dir = os.path.join('./Qi/', dataset, fil_method, '')
    print(dir)
    for i in range(len(dgms)):
        export(dgms[i], dir=dir , filename= str(i) + '.csv')

    print('Finish computing dgms')
    swdgms = dgms2swdgms(dgms)
    kwargs = {'bw': 1, 'K': 1, 'p': 1}  # TODO: K and p is dummy here
    sw_kernel, _ = sw_parallel(swdgms, swdgms, kernel_type='sw', parallel_flag=True, **kwargs)
    clf = classifier(np.zeros((len(labels), 10)), labels, method=None, kernel=sw_kernel)
    print(clf.svm_kernel_())

