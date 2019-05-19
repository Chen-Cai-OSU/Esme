import networkx as nx

def jc(g, edge):
    u, v = edge
    u ,v, p = nx.jaccard_coefficient(g, [(u, v)])
    return p

# g = nx.complete_graph(5)
# jc(g, (1,2))


def add_jaccoef(g):
    tmp = nx.jaccard_coefficient(g, g.edges())
    for u, v, p in tmp:
        g[u][v]['jac'] = p
    return g
