import networkx as nx

def graphforfiltration(g):
    # check fv computation for graph is correct
    for u, v in g.edges():
        try:
            assert g[u][v]['fv'] == max(g.node[u]['fv'], g.node[v]['fv'])
        except AssertionError:
            print('%s, %v is wrong' % (u, v))

def n_min(g):
    res = 0
    valres = []
    attrs = nx.get_node_attributes(g, 'fv')
    for n in g.nodes():
        nbrs = list(nx.neighbors(g, n))
        vals = [attrs[k] for k in nbrs]
        if attrs[n] < min(vals, default=float('inf')):
            res+=1
            valres.append(attrs[n])
    return res, valres
