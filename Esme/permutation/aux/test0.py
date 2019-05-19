from .cycle_tools import timefunction
def getG():
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from([1, 3, 5, 7, 8, 10, 13])
    G.add_edges_from([(1,3), (1,5), (1,7), (5,7), (8,3), (8,5), (10, 5), (10, 7)])
    return G

def summary(G):
    print(G.nodes())
    print(G.edges())

def prob_prefiltration(G, H=0.3, graph_id = 1, sub_flag=True):
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
    summary(G)
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
    old_edges = list(G.edges())  # avoid inplace change
    for s, t in old_edges:
        s_ = list(G.nodes()).index(s);
        t_ = list(G.nodes()).index(t)
        G.add_node(node_idx, edge_p=p_matrix[s_][t_])
        G.add_edges_from([(s, node_idx), (node_idx, t)])
        try:
            G.remove_edge(s, t)
        except:
            pass
        node_idx += 1

    return G

def n_common_nbr(g, e):
    import networkx as nx
    # e = (1,11)
    assert type(e) == tuple
    (u, v) = e
    u_nbr = set(g.neighbors(u))
    v_nbr = set(g.neighbors(v))
    intersection = set.intersection(u_nbr, v_nbr)
    union = set.union(u_nbr, v_nbr)
    return (intersection, union)

def get_node_idx(g, v):
    return list(g.nodes).index(v)

def get_label_similarity_matrix(g, key='label'):
    import numpy as np
    def get_surrounding_key(node_attributes, v, key='label'):
        v = 0
        result = [node_attributes[ky] for ky in list(nx.neighbors(g, v))]
        return result

    def label_similarity(node_attributes, u, v, key='label'):
        res1 = get_surrounding_key(node_attributes, u, key=key)
        res2 = get_surrounding_key(node_attributes, v, key=key)
        return len([x for x in res1 if x in res2])

    import networkx as nx
    similarity_matrix = np.zeros((len(g),len(g)))
    node_attributes = nx.get_node_attributes(g, key)
    for (u,v) in g.edges():
        u_ = get_node_idx(g, u)
        v_ = get_node_idx(g, v)
        similarity_matrix[u_,v_] = label_similarity(node_attributes, u, v, key=key)
    return similarity_matrix
# label_similarity_matrix = get_label_similarity_matrix(g, 'label')


def edge_matrix(G, function_type):
    # a primitive for edge filtration
    # G is a networkx graph
    # function_type can be jaccard, edge_probability, or any user defined function on edges

    import numpy as np
    import networkx as nx
    assert nx.is_connected(G)
    Ga = np.array(nx.to_numpy_matrix(G))
    Gp = np.zeros((len(G), len(G)))
    if function_type[:6] == 'edge_p':
        import sys
        sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/')
        import graphonlib
        try:
            Gp = np.array(graphonlib.smoothing.zhang.smoother(Ga, h=0.3))  # h is a parameter
            assert (Gp == Gp.T).all()
        except:
            print('Graphon Computation Error')
            Gp = np.zeros(np.shape(Gp))
        return (Ga, Gp)

    if function_type[:10] == 'edge_ricci':
        import sys
        sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode')
        from GraphRicciCurvature.OllivierRicci import ricciCurvature
        try:
            G = ricciCurvature(G, alpha=0.5, weight='weight')
        except:
            print('ricciCurvature computation Error from edge matrix')
            for v in G.nodes():
                G[v]['ricciCurvature'] = 0
            for (u, v) in G.edges():
                G[u][v]['ricciCurvature'] = 0

    # if function_type[:10] == 'edge_label':

    for e in G.edges():
        (ints, un) = n_common_nbr(G, e)
        if (function_type[:12] == 'edge_jaccard') and (function_type[:16]!='edge_jaccard_int'):
            sol = len(ints) / float(len(un))
        elif function_type[:16] == 'edge_jaccard_int':
            sol = len(ints)
        elif function_type[:10] == 'edge_ricci':
            # G[e[0]][e[1]]['edge_ricci'] = G[e[0]][e[1]]['ricciCurvature']
            sol = G[e[0]][e[1]]['ricciCurvature']
        else:
            print('Unexpected function_type for edge matrix')
            sol = 0
        e1_ = list(G.nodes()).index(e[0])
        e2_ = list(G.nodes()).index(e[1])
        Gp[e1_][e2_] = sol
        Gp[e2_][e1_] = sol

    return (Ga, Gp)

def edge_filtration(G, average_flag = False, sub_flag = True, function_type='edge_p_min'):
    import numpy as np
    import networkx as nx

    (Ga, Gp) = edge_matrix(G, function_type)
    p_matrix = np.multiply(Ga, Gp)
    assert (p_matrix == p_matrix.T).all()
    if average_flag == True:
        assert function_type[-3:] == 'ave'
    elif average_flag==False:
        assert ((function_type[-3:] == 'min') or (function_type[-3:] == 'max'))

    if average_flag == False:
        if sub_flag ==True:
            assert (function_type[-3:] == 'min')
        elif sub_flag == False:
            assert (function_type[-3:] == 'max')
    name = function_type

    # min-max edge filtration
    for v in G.nodes():
        v_nbr = list(nx.neighbors(G, v))
        v_nbr_ = [list(G.nodes()).index(n) for n in v_nbr]

        v_ = list(G.nodes()).index(v)

        if average_flag==True:
            tmp = [p_matrix[v_][idx] for idx in v_nbr_]
            G.node[v][name] = np.average(tmp)
        elif  sub_flag==True:
            tmp = [p_matrix[v_][idx] for idx in v_nbr_ ]
            G.node[v][name] = min(tmp)
        elif sub_flag == False:
            G.node[v][name] = max(p_matrix[v_])

    # break each edge with the middle point
    node_idx = np.max(G.nodes()) + 1
    old_edges = list(G.edges())  # avoid inplace change
    old_nodes = list(G.nodes())
    for s, t in old_edges:
        s_ = old_nodes.index(s)
        t_ = old_nodes.index(t)
        # print('node idx is', node_idx)
        # print ('s,t is', s,t)
        # print ('s_,t_ is', s_, t_)

        G.add_node(node_idx)
        G.node[node_idx][name] = p_matrix[s_][t_]
        G.node[node_idx]['type']='edge_node'
        G.add_edges_from([(s, node_idx), (node_idx, t)])
        try:
            G.remove_edge(s, t)
        except:
            pass
        node_idx += 1
    return G

def test_table():
    from prettytable import PrettyTable

    x = PrettyTable()

    x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]

    x.add_row(["Adelaide", 1295, 1158259, 600.5])
    x.add_row(["Brisbane", 5905, 1857594, 1146.4])
    x.add_row(["Darwin", 112, 120900, 1714.7])
    x.add_row(["Sydney", 2058, 4336374, 1214.8])
    x.add_row(["Melbourne", 1566, 3806092, 646.9])
    x.add_row(["Perth", 5386, 1554769, 869.4])

    print(x)

def nx2matrix(g):
    import networkx as nx
    from scipy.sparse import csr_matrix
    return csr_matrix(nx.adj_matrix(g)).todense()

def high_order(G):
    import time
    import networkx as nx
    import numpy as np
    t1 = time.time()
    from numpy.linalg import matrix_power
    G2 = nx.power(G,2)
    G3 = nx.power(G,3)
    G4 = nx.power(G,4)
    # G0 = np.identity(len(G))
    # G1 = nx2matrix(G)
    # G2 = matrix_power(G1, 2) - G1 - G0
    # G3 = matrix_power(G1, 3) - G2 - G1 - G0
    # G4 = matrix_power(G1, 4) - G3 - G2 - G1 - G0
    # for x in [G1, G2, G3, G4]:
    #     assert np.min(x) >= 0
    adjlist_2 = nx2matrix(G2)-nx2matrix(G)
    adjlist_3 = nx2matrix(G3) - adjlist_2 - nx2matrix(G)
    adjlist_4 = nx2matrix(G4) - adjlist_3 - adjlist_2 - nx2matrix(G)
    # adjlist_2 = np.sign(G2)
    # adjlist_3 = np.sign(G3)
    # adjlist_4 = np.sign(G4)
    # print adjlist_2, adjlist_3
    assert np.shape(adjlist_2)==np.shape(adjlist_3)==np.shape(adjlist_4) == (len(G), len(G))
    if time.time()-t1>5:
        print(('It takes', time.time()-t1))
    return nx.from_numpy_array(adjlist_2), nx.from_numpy_array(adjlist_3), nx.from_numpy_array(adjlist_4)

# for i in range(2,6):
#     %time nx.power(g, i)



def bad_example():
    #  G.add_node(node_idx)
    # IndexError: index 8 is out of bounds for axis 0 with size 7
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from([0, 33, 2, 1, 6, 7, 34])
    G.add_edges_from([(0, 33), (33, 1), (33, 2), (33, 34), (6, 34), (7, 34)])
    return G

# g = bad_example()
# max(g.nodes())
# edge_filtration(bad_example(), function_type='jaccard')

def test_cache():
    import time
    import numpy as np
    from joblib import Memory
    location = './cachedir'
    memory = Memory(location, verbose=0)
    rng = np.random.RandomState(43)
    data = rng.randn(int(1e5), 10)

    def costly_compute_cached(data, column_index=0):
        """Simulate an expensive computation"""
        time.sleep(5)
        return data[column_index]


    costly_compute_cached = memory.cache(costly_compute_cached)
    start = time.time()
    data_trans = costly_compute_cached(data)
    end = time.time()
    print(('\nThe function took {:.2f} s to compute.'.format(end - start)))
    print(('\nThe transformed data are:\n {}'.format(data_trans)))



if __name__ == '__main__':
    import networkx as nx
    G = getG()
    subgraphs = [G.subgraph(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    g = subgraphs[0].copy()
    g.edges()
    G_ = edge_filtration(g, function_type='edge_jaccard_int_max', average_flag=False, sub_flag=False)
    G_.edges

    for v in G_.nodes():
        print((v, G_.node[v]['edge_jaccard_int_max']))
    for (u,v) in G_.edges():
        print((G_[u][v]['edge_jaccard_int_max']))


