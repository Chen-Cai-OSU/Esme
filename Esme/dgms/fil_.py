""" Implement different filtration stradegies for graphs """

import sys

import dionysus as d
import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from networkx.linalg.algebraicconnectivity import fiedler_vector

from Esme import graphonlib
from Esme.dgms.arithmetic import add_dgm
from Esme.dgms.format import dgm2diag, flip_dgm
from Esme.dgms.stats import dgms_summary, print_dgm, dgm_filter
from Esme.graph.OllivierRicci import ricciCurvature
from Esme.graphonlib.smoothing.zhang import smoother
from Esme.helper.debug import debug
from Esme.helper.load_graph import component_graphs
from Esme.helper.load_graph import load_graphs
from Esme.helper.time import timefunction

class fil_stradegy():
    def __init__(self, g, fil = 'combined', node_fil = 'sub', edge_fil = 'sup', **kwargs):
        """

        :param g: networkx graph
        :param fil: node, edge, or combined
        :param node_fil: used only when fil is node or combined.
        :param edge_fil: used only when fil is edge or combined
        :param kwargs:  kwargs: nodefeat is A np.array of shape (n,1)
                        kwargs: edgefeat is a n*n matrix or other data structure that supports x[u][v]

        """
        self.g = g
        self.n = len(g)
        self.fil = fil
        self.node_fil = node_fil
        self.edge_fil = edge_fil
        self.nodefeat_ = kwargs.get('nodefeat')
        self.edgefeat_ = kwargs.get('edgefeat')

    def nodefeat(self, func = None, nodefeat= None):
        # set node feat for graph

        if func == 'random':
            self.nodefeat_ = np.random.random((self.n, 1))
        else:
            self.nodefeat_ = nodefeat # TODO: should call nodefeat function

    def edgefeat(self, func = None, **kwargs):
        # implement a few common edge vals

        edgefeat = np.zeros((self.n, self.n))
        if func == 'jaccard':
            edgefeat= nx.jaccard_coefficient(self.g, list(self.g.edges))
        elif func == 'edge_p':
            adj = nx.adjacency_matrix(self.g).todense()
            edgefeat = graphonlib.smoothing.zhang.smoother(adj, h=kwargs.get('h', 0.3))  # h : neighborhood size parameter. Example: 0.3 means to include
        else:
            edgefeat = kwargs['edgefeat']

        self.edgefeat_ = edgefeat
        self.edge_fil = kwargs['edge_fil']

    def build_fv(self, **kwargs):
        """
        :param kwargs:
        :return: a networkx where node value and edge value are computed
        """
        if self.fil == 'combined':
            assert self.nodefeat_ is not None
            assert self.edgefeat_ is not None

            for u, v in self.g.edges():
                self.g[u][v]['fv'] = self.edgefeat_[u][v]
            for n in self.g.nodes():
                self.g.node[n]['fv'] = self.nodefeat_[n, 0].astype(float)

        if self.fil == 'node':
            # sublevel fil
            assert self.nodefeat_ is not None
            for n in self.g.nodes():
                self.g.node[n]['fv'] = self.nodefeat_[n, 0].astype(float)

            if self.node_fil == 'sub':
                op = lambda x: max(x)
            elif self.node_fil == 'sup':
                op = lambda x: min(x)
            else:
                raise Exception('Error in node fil')

            for u, v in self.g.edges():
                self.g[u][v]['fv'] = op([self.g.node[u]['fv'], self.g.node[v]['fv']])

        if self.fil == 'edge':
            assert self.edgefeat_ is not None
            for u, v in self.g.edges():
                self.g[u][v]['fv'] = self.edgefeat_[u][v]

            for n in self.g.nodes():
                nbrs = list(nx.neighbors(self.g, n))
                vals = [self.edgefeat_[n][nbr] for nbr in nbrs]
                if len(vals)<1: vals = [0]
                if self.edge_fil == 'sup':
                    self.g.node[n]['fv'] = max(vals)  # min or max? should be max for superlevel and min for sublevel.
                elif self.edge_fil == 'sub':
                    self.g.node[n]['fv'] = min(vals)
                else:
                    raise Exception('Error in edge fil')

        return self.g

class graph2dgm():
    def __init__(self, g, **kwargs):
        self.graph = nx.convert_node_labels_to_integers(g)
        self.nodefil = kwargs.get('nodefil', 'sub')
        self.edgefil = kwargs.get('edgefil', 'sup')

    def check(self):
        for n in self.graph.nodes():
            assert 'fv' in self.graph.node[n]

    def get_simplices(self, gi, key='fv'):
        """Used by get_diagram function"""
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        assert len(gi) > 0
        assert key in gi.node[list(gi.nodes())[0]].keys()
        assert len(gi) == max(list(gi.nodes())) + 1
        simplices = list()
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2][key]):
            tup = ([u, v], data[key])
            simplices.append(tup)
        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            tup = ([v], data[key])
            simplices.append(tup)
        return simplices

    def del_inf(self, dgms):
        # remove inf
        dgms_list = [[], []]
        for i in range(2):
            pt_list = list()
            for pt in dgms[i]:
                if (pt.birth == float('inf')) or (pt.death == float('inf')):
                    pass
                else:
                    pt_list.append(tuple([pt.birth, pt.death]))
            diagram = d.Diagram(pt_list)
            dgms_list[i] = diagram
        return dgms_list

    def compute_PD(self, simplices, sub=True, inf_flag='False', zigzag = False):
        def cmp(a, b):
            return (a > b) - (a < b)
        def compare(s1, s2, sub_flag=True):
            if sub_flag == True:
                if s1.dimension() > s2.dimension():
                    return 1
                elif s1.dimension() < s2.dimension():
                    return -1
                else:
                    return cmp(s1.data, s2.data)
            elif sub_flag == False:
                return -compare(s1, s2, sub_flag=True)
        def zigzag_less(x, y):
            # x, y are simplex
            dimx, datax = x.dimension(), x.data
            dimy, datay = y.dimension(), y.data
            if dimx == dimy == 0:
                return datax <= datay
            elif dimx == dimy == 1:
                return datax >= datay
            else:
                return dimx < dimy

        def zigzag_op(nodefil = 'sub', edgefil = 'sup'):

            if nodefil == 'sub':
                node_op = lambda x, y: x < y
            elif nodefil == 'sup':
                node_op = lambda x, y: x > y
            else:
                raise Exception('node fil Error')

            if edgefil == 'sub':
                edge_op = lambda x, y: x <= y
            elif edgefil == 'sup':
                edge_op = lambda x, y: x > y
            else:
                raise Exception('edge fil Error')

            # x, y are simplex
            def op(x, y):
                dimx, datax = x.dimension(), x.data
                dimy, datay = y.dimension(), y.data
                if dimx == dimy == 0:
                    return node_op(datax, datay)
                elif dimx == dimy == 1:
                    return edge_op(datax, datay)
                else:
                    return dimx < dimy
            return op

        f = d.Filtration()
        for simplex, time in simplices:
            f.append(d.Simplex(simplex, time))

        if not zigzag:
            f.sort() if sub else f.sort(reverse=True)
        else:
            f.sort(zigzag_op(self.nodefil, self.edgefil), reverse=True)
            # print('After zigzag\n')
            # print_f(f)

            # test case
            # simplices = [([2], 4), ([1, 2], 5), ([0, 2], 6),([0], 1), ([1], 2), ([0, 1], 3)]
            # f = d.Filtration()
            # for vertices, time in simplices:
            #     f.append(d.Simplex(vertices, time))
            # f.append(d.Simplex(vertices, time))
            # f.sort(cmp=zigzag_operator(nodefil = 'sub', edgefil = 'sup'),reverse=True)
            # print_f(f)

        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)

        if inf_flag == 'False':
            dgms = self.del_inf(dgms)
        # for some degenerate case, return dgm(0,0)
        if (dgms == []) or (dgms == None):
            return d.Diagram([[0,0]])
        return dgms

    def get_diagram(self, g, key='fv', subflag = 'True', one_homology_flag=False, parallel_flag = False, zigzag = False):
        """

        :param g: networkx graph with fv computed on each node and edge
        :param key: fv. This is the key to access filtration function value
        :param subflag: 'True' if sub level filtration used. 'False' if superlevel filtration used.
        :param one_homology_flag: ignore for now.
        :param parallel_flag: ignore for now.
        :param zigzag: Set to be true if you want to use combined filtration. (set filtration for nodes and edges seprately,
                instead of using node filtration or edge filtration.)
        :return: Persistence diagram
        """

        # only return 0-homology of sublevel filtration TODO: include one homology
        # type can be tuple or pd. tuple can be parallized, pd cannot.
        g = nx.convert_node_labels_to_integers(g)
        simplices = self.get_simplices(g, key = key)
        if one_homology_flag:
            epd_dgm = self.epd(g, pd_flag=False)[1]
            epd_dgm = self.post_process(epd_dgm)
            return epd_dgm

        super_dgms = self.compute_PD(simplices, sub=False)
        sub_dgms = self.compute_PD(simplices, sub=True) if not zigzag else self.compute_PD(simplices, zigzag=True)

        _min = min([g.node[n][key] for n in g.nodes()])
        _max = max([g.node[n][key] for n in g.nodes()])+ 1e-5 # avoid the extra node lies on diagonal
        p_min = d.Diagram([(_min, _max)])
        p_max = d.Diagram([(_max, _min)])

        sub_dgms[0].append(p_min[0])
        super_dgms[0].append(p_max[0])

        if subflag=='True':
            return sub_dgms[0] if not parallel_flag else dgm2diag(sub_dgms[0])
        elif subflag=='False':
            return super_dgms[0] if not parallel_flag else dgm2diag(super_dgms[0])
        else:
            raise Exception('subflag can be either True or False')

    def epd(self, g__, pd_flag=False, debug_flag=False):
        w = -1
        values = nx.get_node_attributes(g__, 'fv')
        simplices = [[x[0], x[1]] for x in list(g__.edges)] + [[x] for x in g__.nodes()]
        up_simplices = [d.Simplex(s, max(values[v] for v in s)) for s in simplices]
        down_simplices = [d.Simplex(s + [w], min(values[v] for v in s)) for s in simplices]
        if pd_flag == True:
            down_simplices = []  # mask the extended persistence here

        up_simplices.sort(key=lambda s1: (s1.dimension(), s1.data))
        down_simplices.sort(reverse=True, key=lambda s: (s.dimension(), s.data))
        f = d.Filtration([d.Simplex([w], -float('inf'))] + up_simplices + down_simplices)
        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)
        if debug_flag == True:
            print('Calling compute_EPD here with success. Print first dgm in dgms')
            print_dgm(dgms[0])
        return dgms

    def post_process(self, dgm, debug_flag=False):
        if len(dgm) == 0:
            return d.Diagram([(0, 0)])
        for p in dgm:
            if p.birth == np.float('-inf'):
                p.birth = 0
            if p.death == np.float('inf'):
                p.death = 0
        if debug_flag == True:
            print('Before flip:'),
            print_dgm(dgm)
        dgm = flip_dgm(dgm)
        if debug_flag == True:
            print('After:'),
            print_dgm(dgm)
        return dgm

#####
def edgefeat(g, norm = False, fil='ricci'):
    """
    wrapper for edge_probability and ricciCurvature computation
    :param g: graph
    :param fil:  edge_p/ricci/jaccard
    :param whether normalize edge values or not
    :return: gp, a dense numpy array of shape (n_node, n_node)
    """
    g = nx.convert_node_labels_to_integers(g)
    assert nx.is_connected(g)
    adj_m = nx.adj_matrix(g).todense() # dense matrix
    gp = np.zeros((len(g), len(g)))
    try:
        if fil == 'edge_p':
            gp = np.array(smoother(adj_m, h=0.3))
            gp = np.multiply(adj_m, gp)
        elif fil == 'ricci':
            g = ricciCurvature(g, alpha=0.5, weight='weight')
            ricci_dict = nx.get_edge_attributes(g, 'ricciCurvature')
            for u, v in ricci_dict.keys():
                gp[u][v] = ricci_dict[(u, v)]
            gp += gp.T
        elif fil == 'jaccard':
            jac_list = nx.jaccard_coefficient(g, g.edges()) # important since jaccard can also be defined on non edge
            for u, v, jac in jac_list:
                gp[u][v] = jac
            gp += gp.T
    except AssertionError:
        print('Have not implemented fil %s. Treat as all zeros'%fil)
        gp = np.zeros((len(g), len(g)))
    assert (gp == gp.T).all()
    if norm: gp = gp / float(max(abs(gp)))
    return gp

def nodefeat(g, fil, norm = False, **kwargs):
    """
    :param g:
    :param fil: deg, cc, random
    :return: node feature (np.array of shape (n_node, 1))
    """
    # g = nx.random_geometric_graph(100, 0.2)
    assert nx.is_connected(g)

    if fil == 'deg':
        nodefeat = np.array(list(dict(nx.degree(g)).values())).reshape(len(g), 1)
    elif fil == 'cc':
        nodefeat = np.array(list(nx.closeness_centrality(g).values()))
        nodefeat = nodefeat.reshape(len(g), 1)
    elif fil == 'random':
        nodefeat = np.random.random((len(g), 1))
    elif fil == 'hop':
        base = kwargs['base']
        assert type(base) == int
        length = nx.single_source_dijkstra_path_length(g, base) # dict #
        nodefeat = [length[i] for i in range(len(g))]
        nodefeat = np.array(nodefeat).reshape(len(g),1)
    elif fil == 'fiedler':
        nodefeat = fiedler_vector(g, normalized=False)  # np.ndarray
        nodefeat = nodefeat.reshape(len(g), 1)
    elif fil == 'ricci':
        g = ricciCurvature(g, alpha=0.5, weight='weight')
        ricci_dict = nx.get_node_attributes(g, 'ricciCurvature')
        ricci_list = [ricci_dict[i] for i in range(len(g))]
        nodefeat = np.array(ricci_list).reshape((len(g), 1))
    else:
        raise Exception('No such filtration: %s'%fil)
    assert nodefeat.shape == (len(g), 1)

    # normalize
    if norm: nodefeat = nodefeat / float(max(abs(nodefeat)))
    return nodefeat

def efeat2nfeat(g, fil='ricci', agg ='min', norm = False):
    """ if edge values is defined, we can sepcify how node value is defined.
    For example agg should be min for sublevel filtration while max for superlevel
    after function.

    :param g: a networkx graph
    :param fil: edge_p/ricci/jaccard
    :param agg: min/max/ave. How edge value is defined from node value
    :return: a node feature of shape (len(g), 1)
    """
    g = nx.convert_node_labels_to_integers(g)
    assert nx.is_connected(g)
    assert agg in ['min', 'max', 'ave']
    if agg == 'min':
        op = lambda x: min(x)
    elif agg == 'max':
        op = lambda x: max(x)
    elif agg == 'ave':
        op = lambda x: np.average(x)
    else:
        raise Exception('Such aggreation %s is not supported'%agg)

    gp = edgefeat(g, fil=fil, norm=norm)
    # print('gp', gp)
    # print('nodes',g.nodes())
    nodefeat = []
    for v in g.nodes():
        v_nbr = list(nx.neighbors(g,v))
        v_list = [gp[v][nbr] for nbr in v_nbr]
        nodefeat.append(op(v_list))
    nodefeat = np.array(nodefeat).reshape(len(g),1)
    return nodefeat

######
def node_fil_(g = None, fil = 'deg', fil_d = 'sub', norm = False, one_hom = False, **kwargs):
    """
    sublevel filtration

    :param g: graph
    :param fil: filtration type. Deg, cc, ricciCurvature where fv is normalized.
    :param fil_d: filtration direction
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
    fil = fil_stradegy(g, fil='node', node_fil=fil_d, nodefeat=nodefeat_)
    g = fil.build_fv()

    x = graph2dgm(g)
    subflag = 'True' if fil_d == 'sub' else 'False'
    diagram = x.get_diagram(g, key='fv', subflag=subflag, one_homology_flag=one_hom, parallel_flag=False, zigzag=False)
    return diagram

def edge_fil_(g = None, fil='jaccard', fil_d = 'sub', norm = False, one_hom = False, **kwargs):
    """
    :param g: networkx graph
    :param fil: edge filtration: edge_p/ricci/jaccard
    :param fil_d: sub/sup
    :param norm: noramlize edge feat or not
    :param one_hom: whether get 1-homology diagram
    :param kwargs: not used
    :return: diagram
    """
    g = nx.convert_node_labels_to_integers(g)
    agg = 'min' if fil_d == 'sub' else 'max'
    nodefeat_ = efeat2nfeat(g, fil=fil, agg=agg, norm=norm)
    fil = fil_stradegy(g, fil='node', node_fil=fil_d, nodefeat=nodefeat_) # treat this as node fil
    g = fil.build_fv()

    x = graph2dgm(g)
    subflag = 'True' if fil_d == 'sub' else 'False'
    diagram = x.get_diagram(g, key='fv', subflag=subflag, one_homology_flag=one_hom, parallel_flag=False, zigzag=False)
    return diagram

######
def g2dgm(i, g=None, fil='deg', fil_d = 'sub', norm=False, one_hom=False, debug_flag = False, **kwargs):
    """
    a wrapper of node_fil_ for parallel computing dgms.
    :param g:
    :param fil:
    :param fil_d: sub/super
    :param norm: False by default
    :param one_hom: False by default
    :param debug_flag: False by default
    :param kwargs:
    :return:
    """
    # assert 'gs' in globals().keys()
    # g = gs[i].copy()
    if debug_flag:
        print('processing %s-th graph where fil is %s and fil_d is %s' % (i, fil, fil_d))

    components = component_graphs(g)
    dgm = d.Diagram([])
    for component in components:
        if fil in ['jaccard', 'ricci', 'edge_p']:
            tmp_dgm = edge_fil_(component, fil=fil, fil_d=fil_d, norm=norm, one_hom=one_hom, **kwargs)
            print_dgm(tmp_dgm)
        else:
            tmp_dgm = node_fil_(component, fil=fil, fil_d=fil_d, norm=norm, one_hom=one_hom, **kwargs)
        dgm = add_dgm(dgm, tmp_dgm)
        dgm = dgm_filter(dgm)
    dgm = dgm_filter(dgm) # handle the case when comonents is empty
    return dgm

def gs2dgms_parallel(n_jobs = 1, **kwargs):
    global gs
    try:
        assert 'gs' in globals().keys()
    except AssertionError:
        print(globals().keys())

    dgms = Parallel(n_jobs=n_jobs)(delayed(g2dgm)(i, g=gs[i], **kwargs) for i in range(len(gs)))
    return dgms

@timefunction
def gs2dgms(gs, fil='deg', fil_d = 'sub', norm=False, one_hom=False, debug_flag = False, **kwargs):
    """
    serial computing dgms
    :param gs: a list of nx graphs
    :param fil: filtration(deg, ricci)
    :param fil_d : sub or sup
    :param norm: whether normalize or not
    :param one_hom: one homology or not
    :param debug_flag: False by default
    :return: dgms: a list of dgm
    """
    dgms = []
    for i in range(len(gs)):
        if debug_flag:
            print('processing %s-th graph where fil is %s and fil_d is %s'%(i, fil, fil_d))
        components = component_graphs(gs[i])
        if len(components)==0: return d.Diagram([[0,0]])

        dgm = d.Diagram([])
        for component in components:
            tmp_dgm = node_fil_(component, fil=fil, fil_d = fil_d, norm=norm, one_hom=one_hom, **kwargs)
            dgm = add_dgm(dgm, tmp_dgm)
            dgm = dgm_filter(dgm)
        assert len(dgm) > 0
        dgms.append(dgm)

    return dgms

if __name__ == '__main__':
    g = nx.random_geometric_graph(1000, 0.1, seed=42)

    # combined fil -- one way
    fil = fil_stradegy(g, fil='combined', nodefeat=np.random.random((len(g), 1)))
    fil.edgefeat(func='edge_p', edge_fil='sup')
    g1 = fil.build_fv()
    x = graph2dgm(g1, nodefil='sub', edgefil='sup')
    diagram = x.get_diagram(g1, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=True)
    print(diagram)

    # combined fil -- anathor way. Almost the same with the previous example.
    fil = fil_stradegy(g, fil='combined')
    fil.edgefeat(func='edge_p', edge_fil='sup')
    fil.nodefeat(func='random')
    g2 = fil.build_fv()
    x = graph2dgm(g2, nodefil='sub', edgefil='sup')
    dgm = x.get_diagram(g2, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=True)
    print_dgm(dgm)
    sys.exit()

    # edge feat example
    fil = fil_stradegy(g, fil='edge')
    fil.edgefeat(func='edge_p', edge_fil='sup')
    g = fil.build_fv()
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=False)
    print(diagram)

    # node feat example
    nodefeat_ = np.array(list(dict(nx.degree(g)).values())).reshape(len(g), 1)  # np.random.random((n_node, 1))
    nodefeat_ = nodefeat_ / float(max(nodefeat_))
    fil = fil_stradegy(g, fil='node', node_fil='sub', nodefeat=nodefeat_)
    g = fil.build_fv()
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=False)
    print(diagram)

    # imdb
    gs, labels = load_graphs(dataset='imdb_binary')  # step 1
    subdgms = gs2dgms_parallel(n_jobs=1, fil='jaccard', fil_d='sub', one_hom=False, debug_flag=False)  # step2 # TODO: need to add interface
    dgms_summary(subdgms)
    debug(subdgms, 'subdgms')

    g = nx.random_geometric_graph(100, 0.4)
    print(edgefeat(g, fil='jaccard'))
    np.random.seed(42)
    n_node = 20
    g = nx.random_geometric_graph(n_node, 0.5, seed=42)
    diagram = node_fil_(g, fil = 'hop', norm=True, base=0)
    print(diagram)

