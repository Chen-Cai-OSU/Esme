""" some tmp functions """
import networkx as nx
from Esme.dgms.fil import nodefeat, fil_stradegy, graph2dgm
def node_fil(g = None, fil = 'deg', norm = False, one_hom = False, **kwargs):
    """
    sublevel filtration
    :param g: graph
    :param fil: filtration type. Deg, cc, ricciCurvature where fv is normalized.
    :param norm: whether normalize fv
    :param kwargs: i: index of global gs
    :return: Persistence diagram.
    """
    # if g is None:
    #     assert 'gs' in globals().keys()
    #     i = kwargs['i']
    #     g = gs[i]

    g = nx.convert_node_labels_to_integers(g)
    nodefeat_ = nodefeat(g, fil, norm=norm, **kwargs)
    fil = fil_stradegy(g, fil='node', node_fil='sub', nodefeat=nodefeat_)
    g = fil.build_fv()

    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=one_hom, parallel_flag=False, zigzag=False)
    return diagram
