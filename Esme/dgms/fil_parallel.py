import networkx as nx
from joblib import Parallel, delayed

from Esme.dgms.fil import nodefeat, fil_stradegy, graph2dgm
from Esme.helper.time import timefunction

# @timefunction
def parallel_cg(n_jobs=-1):
    # g = nx.random_geometric_graph(100, 0.2)
    result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(node_fil)(gs[i]) for i in range(10, 50))
    # print(result)

def gs():
    gs = []
    for i in range(100):
        g = nx.random_geometric_graph(100, 0.5)
        gs.append(g)
    return gs

def node_fil(g = None, fil = 'deg', norm = False, one_hom = False, **kwargs):
    """
    :param g: graph
    :param fil: filtration type. Deg, cc, ricciCurvature where fv is normalized.
    :param norm: whether normalize fv
    :param kwargs: i: index of global gs
    :return: Persistence diagram.
    """
    if g is None:
        assert 'gs' in globals().keys()
        i = kwargs['i']

    g = gs[kwargs['i']]
    nodefeat_ = nodefeat(g, fil, norm=norm, **kwargs)
    fil = fil_stradegy(g, fil='node', node_fil='sub', nodefeat=nodefeat_)
    g = fil.build_fv()
    for u, v in g.edges():
        assert g[u][v]['fv'] == max(g.node[u]['fv'], g.node[v]['fv'])

    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=one_hom, parallel_flag=False, zigzag=False)
    if kwargs['debug']:
        print('Finish %s'%kwargs['i'])
    return diagram


if __name__ == '__main__':
    gs = gs()

    # print('serial')
    # parallel_cg(n_jobs=1)

    print('parallel')
    # parallel_cg(n_jobs=-1)
    g = gs[1]
    Parallel(n_jobs=-1, verbose=1)(delayed(node_fil)(i=i, debug=True) for i in range(10, 50))