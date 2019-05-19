import networkx as nx
import numpy as np
np.random.seed(42)
import Esme.graphonlib as graphonlib
from Esme.viz.matrix import viz_matrix

def function_basis(g, allowed, norm_flag = 'no'):
    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    # allowed = ['ricci', 'deg', 'hop', 'cc', 'fiedler']
    # to save recomputation. Look at the existing feature at first and then simply compute the new one.

    if len(g)<3: return
    assert nx.is_connected(g)

    def norm(g, key, flag=norm_flag):
        if flag=='no':
            return 1
        elif flag == 'yes':
            res = nx.get_node_attributes(g, key).values()
            return np.max(np.abs(list(nx.get_node_attributes(g, key).values()))) + 1e-6

    if 'deg' in allowed:
        deg_dict = dict(nx.degree(g))
        for n in g.nodes():
            g.node[n]['deg'] = deg_dict[n]
        deg_norm = norm(g, 'deg', norm_flag)
        for n in g.nodes():
            g.node[n]['deg'] /= np.float(deg_norm)
    elif 'random' in allowed:
        for n in g.nodes():
            g.node[n]['random'] = np.random.rand()


    if 'deg' in allowed:
        for n in g.nodes():
            attribute_mean(g, n, key='deg', cutoff=1, iteration=0)
        if norm_flag == 'yes':
            # better normalization
            for attr in [ '1_0_deg_sum']: # used to include 1_0_deg_std/ deleted now:
                norm_ = norm(g, attr, norm_flag)
                for n in g.nodes():
                    g.node[n][attr] /= float(norm_)
    return g

def add_edge_val(gi, edge_value='max'): # only works for sublevel filtration
    for (e1, e2) in gi.edges():
        if edge_value == 'max':
            gi[e1][e2]['fv'] = max(gi.node[e1]['fv'], gi.node[e2]['fv'])
        if edge_value == 'min':
            gi[e1][e2]['fv'] = min(gi.node[e1]['fv'], gi.node[e2]['fv'])

def add_edgeval(g, fil='deg'):
    """  use this after calling function_basis. """
    for n in g.nodes():
        g.node[n]['fv'] = g.node[n][fil]
    add_edge_val(g, edge_value='max')
    return g

def attribute_mean(g, i, key='deg', cutoff=1, iteration=0):
    # g = graphs_[i][0]
    # g = graphs_[0][0]
    # attribute_mean(g, 0, iteration=1)
    for itr in [iteration]:
        assert key in g.node[i].keys()
        # nodes_b = nx.single_source_shortest_path_length(g,i,cutoff=cutoff).keys()
        # nodes_a = nx.single_source_shortest_path_length(g,i,cutoff=cutoff-1).keys()
        # nodes = [k for k in nodes_b if k not in nodes_a]
        nodes = g[i].keys()

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

        if len(nbrs_deg) == 0:
            g.node[i][key_mean] = 0
            g.node[i][key_min] = 0
            g.node[i][key_max] = 0
            g.node[i][key_std] = 0
            g.node[i][key_sum] = 0
        else:
            # assert np.max(nbrs_deg) < 1.1
            g.node[i][key_mean] = np.mean(nbrs_deg)
            g.node[i][key_min] = np.min(nbrs_deg)
            g.node[i][key_max] = np.max(nbrs_deg)
            g.node[i][key_std] = np.std(nbrs_deg)
            g.node[i][key_sum] = np.sum(nbrs_deg)

def fil_strategy(g, nodefeat, method='node', viz_flag = False, **kwargs):
    """ given a graph, implement node/edge and combined filtration
        kwargs are used for edge filtration
    """
    # g = nx.random_geometric_graph(100, 0.3)
    g = nx.convert_node_labels_to_integers(g)
    a = nx.adjacency_matrix(g).todense()

    if method=='edge':
        if kwargs['edgefunc'] == 'edge_prob':
            try:
                p_zhang = graphonlib.smoothing.zhang.smoother(a, h=kwargs['h'])  # h : neighborhood size parameter. Example: 0.3 means to include
            except ValueError: # for nci1, some adj matrix is rather small
                print('Exception: set p_zhang as 0')
                p_zhang = np.zeros((len(g), len(g)))
            assert p_zhang.shape == (len(g), len(g))
            if viz_flag: viz_matrix(p_zhang)
            for u, v in g.edges():
                g[u][v]['fv'] = p_zhang[u][v]

        elif kwargs['edgefunc'] == 'jaccard':
            jaccard = nx.jaccard_coefficient(g, list(g.edges))
            jaccard_m = np.zeros((len(g), len(g)))
            for u, v, p in jaccard:
                jaccard_m[u][v] = p
            for u, v in g.edges():
                g[u][v]['fv'] = jaccard_m[u][v]
        else:
            raise Exception('No such edgefunc %s implemented '%kwargs['edgefunc'])

        attributes = nx.get_edge_attributes(g, 'fv')
        for n in g.nodes():
            nbrs = nx.neighbors(g, n)
            keys = [key for key in attributes.keys() if n in key]
            vals = [attributes[key] for key in keys]
            if vals == []: vals = [0]
            g.node[n]['fv'] = max(vals)  # min or max? should be max for superlevel and min for sublevel!

    elif method=='node':
        for n in g.nodes():
            g.node[n]['fv'] = nodefeat[n, 0].astype(float)
        for u, v in g.edges():
            g[u][v]['fv'] = max(g.node[u]['fv'], g.node[v]['fv'])

    elif method == 'combined':
        # need to turn on zigzag flag
        try:
            p_zhang = graphonlib.smoothing.zhang.smoother(a, h=kwargs['h'])  # h : neighborhood size parameter. Example: 0.3 means to include
        except ValueError:  # for nci1, some adj matrix is rather small
            print('Exception: set p_zhang as 0')
            p_zhang = np.zeros((len(g), len(g)))

        for u, v in g.edges():
            g[u][v]['fv'] = p_zhang[u][v]
        for n in g.nodes():
            g.node[n]['fv'] = nodefeat[n, 0].astype(float)

    else:
        raise Exception('No such filtration strategy')

    return g
