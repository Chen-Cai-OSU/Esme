import matplotlib.pyplot as plt
import pygsp
from Esme.viz.graph import viz_graph
import networkx as nx
from networkx import stochastic_block_model

def sbm(n=1000, p=0.01):
    sizes = [n] * 3
    labels = [1] * n + [2] * n + [3] * n
    probs = [[0.5, p, p],
             [p, 0.5, p],
             [p, p, 0.5]]
    g = stochastic_block_model(sizes, probs, seed=42)
    return g, labels

def sbm2(n1=100, n2=50, p = 0.5, q=0.1):
    sizes = [n1, n2]
    labels = [1] * n1 + [2] * n2
    probs = [[p, q],[q, p]]
    g = stochastic_block_model(sizes, probs, seed=42)
    return g, labels

def sbms(n = 100, n1=100, n2=50, p = 0.5, q=0.1):
    # generate n sbm graphs
    gs = []
    sizes = [n1, n2]
    probs = [[p, q], [q, p]]
    for i in range(n):
        g = stochastic_block_model(sizes, probs, seed=i)
        gs.append(g)
    return gs

def sbm3(n=1000, p=0.01):
    sizes = [n] * 3
    labels = [1] * n + [2] * n + [3] * n
    probs = [[0.5, p, p],
             [p, 0.5, p],
             [p, p, 0.5]]
    g = stochastic_block_model(sizes, probs, seed=42)
    return g, labels

def sbms3(n = 100, n1=100, n2=50, n3=50, p = 0.5, q=0.1):
    # generate n sbm graphs
    gs = []
    sizes = [n1, n2, n3]
    probs = [[p, q, q], [q, p, q], [q, q, p]]
    for i in range(n):
        g = stochastic_block_model(sizes, probs, seed=i)
        gs.append(g)
    return gs

def swiss_roll_graph():
    G = pygsp.graphs.SwissRoll(N=200, seed=42)
    g = nx.from_scipy_sparse_matrix(G.W)
    viz_graph(g, show=True)
    return g

def rgg(n, d, radius=1):
    return nx.random_geometric_graph(n, radius=1, dim=d)

def iso_graphs(version=2):
    # Return the list of all graphs with up to seven nodes named in the Graph Atlas.
    if version == 1:
        graphs = []
        n = len(nx.graph_atlas_g())
        for i in range(n):
            graphs.append([nx.graph_atlas(i)])
        return graphs
    elif version == 2:
        graphs = nx.graph_atlas_g()
        graphs = [[g] for g in graphs]
        return graphs[1:]

