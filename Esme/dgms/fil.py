""" Implement different filtration stradegies for graphs """

import sys
import time
import dionysus as d
import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from networkx.linalg.algebraicconnectivity import fiedler_vector

from Esme import graphonlib
from Esme.dgms.arithmetic import add_dgm
from Esme.dgms.format import dgm2diag, flip_dgm
from Esme.dgms.stats import dgms_summary
from Esme.dgms.stats import print_dgm
from Esme.graph.OllivierRicci import ricciCurvature
from Esme.graphonlib.smoothing.zhang import smoother
from Esme.helper.debug import debug
from Esme.helper.load_graph import component_graphs
from Esme.helper.load_graph import load_graphs
from Esme.helper.time import timefunction, time_node_fil
from Esme.dgms.fake import array2dgm
from Esme.dgms.ioio import dgms_dir_test, load_dgms, save_dgms
from Esme.graph.dataset.modelnet import modelnet2graphs
from Esme.dgms.format import export_dgm
from Esme.helper.dgms import check_single_dgm
import os
from Esme.helper.dgms import check_partial_dgms

np.random.seed(42)

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
            self.nodefeat_ = np.random.random((self.g, 1))
        else:
            self.nodefeat_ = nodefeat

    def edgefeat(self, func = None, **kwargs):
        # implement a few common edge vals

        edgefeat = np.zeros((self.n, self.n))
        if func == 'jaccard':
            edgefeat= nx.jaccard_coefficient(self.g, list(self.g.edges))
        elif func == 'edge_prob':
            adj = nx.adjacency_matrix(self.g).todense()
            edgefeat = graphonlib.smoothing.zhang.smoother(adj, h=kwargs.get('h', 0.3))  # h : neighborhood size parameter. Example: 0.3 means to include
        else:
            edgefeat = kwargs['edgefeat']

        self.edgefeat_ = edgefeat
        self.edge_fil = kwargs['edge_fil']

    def build_fv(self, **kwargs):
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

    @timefunction
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

def _edgefeat(g, norm = False, fil='ricci'):
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

def edgefeat(g, fil='ricci', agg = 'min', norm = False):
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

    gp = _edgefeat(g, fil=fil, norm=norm)
    # print('gp', gp)
    # print('nodes',g.nodes())
    nodefeat = []
    for v in g.nodes():
        v_nbr = list(nx.neighbors(g,v))
        v_list = [gp[v][nbr] for nbr in v_nbr]
        nodefeat.append(op(v_list))
    nodefeat = np.array(nodefeat).reshape(len(g),1)
    return nodefeat

@timefunction
def nodefeat(g, fil, norm = False, **kwargs):
    """
    :param g:
    :param fil: deg, cc, random
    :return: node feature (np.array of shape (n_node, 1))
    """
    # g = nx.random_geometric_graph(100, 0.2)
    t0 = time.time()
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
        if len(g.edges) == 2* len(g): # todo hack here. fielder is very slow when n_edges = 2*n_edge
            nodefeat = np.array(list(dict(nx.degree(g)).values())).reshape(len(g), 1)
        else:
            nodefeat = fiedler_vector(g, normalized=False)  # np.ndarray
            nodefeat = nodefeat.reshape(len(g), 1)

    elif fil == 'fiedler_w':
        if False: # len(g.edges) == 2 * len(g):  # todo hack here. fielder is very slow when n_edges = 2*n_edge
            nodefeat = np.array(list(dict(nx.degree(g)).values())).reshape(len(g), 1)
        else:
            for u,v in g.edges():
                try:
                    assert 'dist' in g[u][v].keys()
                    g[u][v]['dist'] += 1e-6
                except AssertionError:
                    pass
                    # print(f'g[{u}][{v}] = {g[u][v]}')
            print(f'bottleneck graph {len(g)}/{len(g.edges())}')
            # for line in nx.generate_edgelist(g):
            #     print(line)
            print('-'*50)
            nodefeat = fiedler_vector(g, normalized=False, weight='dist', method='tracemin_lu')  # np.ndarray
            print('after true fiedler')
            nodefeat = nodefeat.reshape(len(g), 1)


    elif fil == 'ricci':
        try:
            g = ricciCurvature(g, alpha=0.5, weight='weight')
            ricci_dict = nx.get_node_attributes(g, 'ricciCurvature')
            ricci_list = [ricci_dict[i] for i in range(len(g))]
            nodefeat = np.array(ricci_list).reshape((len(g), 1))
        except:
            nodefeat = np.random.random((len(g),1)) # cvxpy.error.SolverError: Solver 'ECOS' failed. Try another solver.
    elif fil[:3] == 'hks':
        assert fil[3] == '_'
        t = float(fil[4:])
        from Esme.dgms.hks import hks
        nodefeat = hks(g, t)
    else:
        raise Exception('No such filtration: %s'%fil)
    assert nodefeat.shape == (len(g), 1)


    # normalize
    if norm: nodefeat = nodefeat / float(max(abs(nodefeat)))
    if time.time()-t0 >3:
        from Esme.helper.time import precision_format
        print(f'nodefeat takes {precision_format(time.time()-t0, 2)} for g {len(g)}/{len(g.edges)}')
        from Esme.viz.graph import viz_graph
        # viz_graph(g, show=True)
    return nodefeat

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

# @time_node_fil
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
    # print('in node_fil_', kwargs)

    g = nx.convert_node_labels_to_integers(g)
    nodefeat_ = nodefeat(g, fil, norm=norm, **kwargs)
    assert kwargs.get('ntda', None) in [True, False, None]
    if kwargs.get('ntda', None) == True: return array2dgm(nodefeat_, fil_d = fil_d)

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
    nodefeat_ = edgefeat(g, fil=fil, agg=agg, norm=norm)
    fil = fil_stradegy(g, fil='node', node_fil=fil_d, nodefeat=nodefeat_) # treat this as node fil
    g = fil.build_fv()

    x = graph2dgm(g)
    subflag = 'True' if fil_d == 'sub' else 'False'
    diagram = x.get_diagram(g, key='fv', subflag=subflag, one_homology_flag=one_hom, parallel_flag=False, zigzag=False)
    return diagram


def g2dgm(i, g=None, fil='deg', fil_d = 'sub', norm=False, one_hom=False, debug_flag = False, **kwargs):
    """
    a wrapper of node_fil_ for parallel computing dgms.
    :param g:
    :param fil:
    :param fil_d:
    :param norm: False by default
    :param one_hom: False by default
    :param debug_flag: False by default
    :param kwargs:
    :return:
    """
    # assert 'gs' in globals().keys()
    # g = gs[i].copy()

    if len(g) > 60000:
        return d.Diagram([[0,0]]) # todo better handling

    if debug_flag:
        print('in g2dm', kwargs)
        i += kwargs.get('a', 0)
        print(f'processing {i}-th graph({len(g)}/{len(g.edges)}) where fil is {fil} and fil_d is {fil_d} and one_hom is {one_hom}')

    if kwargs['write'] == True:  # 一个后门
        fil_d_ = 'epd' if one_hom == True else  fil_d
        if check_single_dgm(graph = 'mn'+version, fil = fil, fil_d=fil_d_, norm=norm, idx=i): return


    components = component_graphs(g)
    dgm = d.Diagram([])
    for component in components:
        if fil in ['jaccard']:
            tmp_dgm = edge_fil_(component, fil=fil, fil_d=fil_d, norm=norm, one_hom=one_hom, **kwargs)
            print_dgm(tmp_dgm)
        else:
            tmp_dgm = node_fil_(g = component, fil=fil, fil_d=fil_d, norm=norm, one_hom=one_hom, **kwargs)

        dgm = add_dgm(dgm, tmp_dgm)
        dgm = dgm_filter(dgm)
    dgm = dgm_filter(dgm) # handle the case when comonents is empty

    if kwargs['write'] == True: # 一个后门
        if one_hom == True: fil_d = 'epd'
        dir = os.path.join('/home/cai.507/anaconda3/lib/python3.6/site-packages/save_dgms/', 'mn' + version, fil, fil_d,'norm_' + str(norm), '')
        export_dgm(dgm, dir=dir, filename=  str(i) +'.csv', print_flag=True)
    return dgm


def gs2dgms_parallel(n_jobs = 1, **kwargs):
    """ a wraaper of g2dgm for parallel computation
        sync with the same-named function in fil.py
        put here for parallelizaiton reason
    """

    if dgms_dir_test(**kwargs)[1]: #and kwargs.get('ntda', None)!=True: # load only when ntda=False
        dgms = load_dgms(**kwargs)
        return dgms
    try:
        assert 'gs' in globals().keys()
    except AssertionError:
        print(globals().keys())

    try:
        # print('in gs2dgms_parallel', kwargs)
        dgms = Parallel(n_jobs=n_jobs)(delayed(g2dgm)(i, gs[i], **kwargs) for i in range(len(gs)))
    except NameError:  # name gs is not defined
        sys.exit('NameError and exit')

    save_dgms(dgms, **kwargs)
    return dgms

def _gs2dgms_parallel(n_jobs = 1, **kwargs):
    """ a wraaper of g2dgm for parallel computation """
    if dgms_dir_test(**kwargs)[1]:
        dgms = load_dgms(**kwargs)
        return dgms
    try:
        assert 'gs' in globals().keys()
    except AssertionError:
        print(globals().keys())

    try:
        dgms = Parallel(n_jobs=n_jobs)(delayed(g2dgm)(i, gs[i], **kwargs) for i in range(len(gs)))
    except NameError:  # name gs is not defined
        print('NameError and exit')
        sys.exit()

    save_dgms(dgms, **kwargs)
    return dgms


@timefunction
def gs2dgms(gs, fil='deg', fil_d = 'sub', norm=False, one_hom=False, debug_flag = False, **kwargs):
    """
    serial computing dgms
    :param gs: a list of raw nx graphs(no function value)
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
            print(f'process {i}-th graph({len(gs[i])}/{len(nx.edges(gs[i]))}) where one_hom is {one_hom} fil is {fil} and fil_d is {fil_d}')

        components = component_graphs(gs[i]) # todo chnage back to 4
        # components4 = component_graphs(gs[i], threshold=4) #todo
        # components5 = component_graphs(gs[i], threshold=5) #todo

        # print(f'threshold 4/5 has {len(components4)}/{len(components5)}')
        if len(components)==0: return d.Diagram([[0,0]])

        dgm = d.Diagram([])
        for component in components:
            tmp_dgm = node_fil_(g = component, fil=fil, fil_d = fil_d, norm=norm, one_hom=one_hom, **kwargs)
            dgm = add_dgm(dgm, tmp_dgm)
            dgm = dgm_filter(dgm)
            # TODO: implement edge_fil_
        assert len(dgm) > 0
        dgms.append(dgm)

    return dgms

def dgm_filter(dgm):
    """ if input is an empyt dgm, add origin point """
    if len(dgm) > 0:
        return dgm
    else:
        return d.Diagram([[0,0]])

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--a", default=0, type=int, help='write files in batch. a is beginner')
parser.add_argument("--b", default=100, type=int, help='write files in batch. b is ender')
parser.add_argument("--n_jobs", default=1, type=int, help='n_jobs')
parser.add_argument("--step_size", default=1, type=int, help='step size for sampling')

parser.add_argument("--parallel", action='store_true', help='use parallel or not')
parser.add_argument("--sub", action='store_true', help='compute sub only')
parser.add_argument("--sup", action='store_true', help='compute sup only')
parser.add_argument("--epd", action='store_true', help='compute epd only')
parser.add_argument("--all", action='store_true', help='compute epd only')

if __name__ == '__main__':
    # g = nx.random_geometric_graph(100, 0.3)
    # nodefeat = edgefeat(g, fil='ricci', agg='min', norm=False)
    # print(nodefeat.shape)

    # compute for graphs that are larger than memory
    args = parser.parse_args()
    version = '10'
    a, b, n_jobs = args.a, args.b, args.n_jobs
    fil = 'fiedler_w' # 'fiedler'

    kw = {'a':a, 'b':b, 'fil': fil}
    if check_partial_dgms(**kw, fil_d='sub') and check_partial_dgms(**kw, fil_d='sup') and check_partial_dgms(**kw, fil_d='epd') : sys.exit()

    gs, labels = modelnet2graphs(version=version, print_flag=True, a = a, b = b, weight_flag=True) # todo weight_flag is false for fiedler
    norm, ntda = True, False

    for fil in [fil]: # ['hks_1']: #['hks_1', 'hks_10', 'hks_0.1']:

        if args.parallel:
            kwargs = {'fil': fil, 'fil_d': 'sub', 'norm': norm, 'graph': 'mn' + version, 'ntda': ntda, 'debug_flag': True}

            kwargs['write'] = True # for testing the new way for parallem computing
            kwargs['a'] = a

            if args.sub or args.all:
                subdgms = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(g2dgm)(i, gs[i], **kwargs) for i in range(0, len(gs),args.step_size))
                del subdgms

            if args.sup or args.all:
                kwargs['fil_d'] = 'sup'
                supdgms = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(g2dgm)(i, gs[i], **kwargs) for i in range(0, len(gs),args.step_size))
                del supdgms

            if args.epd or args.all:
                kwargs['one_hom'] = True
                epddgms = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(g2dgm)(i, gs[i], **kwargs) for i in range(0, len(gs),args.step_size))
                del epddgms

        else:
            # subdgms = gs2dgms(gs, n_jobs=n_jobs, fil=fil, fil_d='sub', norm=norm, graph = 'mn' + version, ntda = ntda, debug_flag = True)
            supdgms = gs2dgms(gs, n_jobs=n_jobs, fil=fil, fil_d='sup', norm=norm, graph='mn' + version, ntda=ntda, debug_flag=True)
            # epddgms = gs2dgms(gs, fil=fil, one_hom=True, norm=norm, graph='mn' + version, ntda=ntda, debug_flag=True, n_jobs=n_jobs)

        continue
        for i in range(a,b):
            dir = os.path.join('/home/cai.507/anaconda3/lib/python3.6/site-packages/save_dgms/mn10/', fil, 'sub/norm_True/' )
            export_dgm(subdgms[i-a], dir=dir, filename= str(i) +'.csv')

            dir = os.path.join('/home/cai.507/anaconda3/lib/python3.6/site-packages/save_dgms/mn10/', fil, 'sup/norm_True/')
            export_dgm(supdgms[i - a], dir=dir, filename=str(i) + '.csv')

            dir = os.path.join('/home/cai.507/anaconda3/lib/python3.6/site-packages/save_dgms/mn10/', fil, 'epd/norm_True/')
            export_dgm(epddgms[i - a], dir=dir, filename=str(i) + '.csv')

    sys.exit()



    # computing dgms for replicate.py
    graph = 'cox2'
    for graph in  ['bzr', 'cox2', 'dhfr', 'dd_test', 'nci1',  'frankenstein', 'protein_data',   'imdb_binary',  'imdb_multi', 'reddit_binary', 'reddit_5K']:
        from Esme.graph.dataset.tu_dataset import load_tugraphs, load_shapegraphs
        from Esme.graph.dataset.modelnet import modelnet2graphs
        gs, labels = load_tugraphs(graph)
        # gs, labels = modelnet2graphs(version='10', print_flag=True, test_size=None) # load_shapegraphs(graph)
        norm = True # todo: change to True
        for fil in ['hks_1', 'hks_10', 'hks_0.1']: # ['deg', 'random', 'cc', 'fiedler', 'ricci']:
            n_jobs = 1
            for ntda in [False]:
                # subdgms = gs2dgms(gs=gs, n_jobs=n_jobs, fil=fil, fil_d='sub', norm=norm, graph = graph, ntda = ntda, debug_flag = True)
                # supdgms = gs2dgms(gs=gs, n_jobs=n_jobs, fil=fil, fil_d='sup', norm=norm, graph = graph, ntda = ntda, debug_flag = True)
                # epddgms = gs2dgms(gs=gs, n_jobs=n_jobs, fil=fil, one_hom=True, norm=norm, graph = graph, ntda = ntda, debug_flag = True)

                subdgms = gs2dgms_parallel(n_jobs=n_jobs, fil=fil, fil_d='sub', norm=norm, graph = graph, ntda = ntda, debug_flag = True)
                supdgms = gs2dgms_parallel(n_jobs=n_jobs, fil=fil, fil_d='sup', norm=norm, graph = graph, ntda = ntda, debug_flag = True)
                epddgms = gs2dgms_parallel(n_jobs=n_jobs, fil=fil, one_hom=True, norm=norm, graph = graph, ntda = ntda, debug_flag = True)
        continue
        for fil in ['deg', 'random', 'ricci', 'cc', 'fiedler']:
            for ntda in [True]:
                subdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sub', norm=norm, graph=graph, ntda=ntda)
                supdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sup', norm=norm, graph=graph, ntda=ntda)
                epddgms = gs2dgms_parallel(n_jobs=-1, fil=fil, one_hom=True, norm=norm, graph=graph, ntda=ntda)

    sys.exit()

    save_dgms(subdgms, **kwargs)
    dgms_summary(subdgms)
    debug(subdgms, 'subdgms')


    g = nx.random_geometric_graph(100, 0.4)
    print(edgefeat(g, fil='jaccard'))
    np.random.seed(42)
    n_node = 20
    g = nx.random_geometric_graph(n_node, 0.5, seed=42)
    diagram = node_fil(g, fil = 'hop', norm=True, base=0)
    print(diagram)

    # node feat example
    nodefeat = np.array(list(dict(nx.degree(g)).values())).reshape(len(g),1) # np.random.random((n_node, 1))
    nonfeat = nodefeat / float(max(nodefeat))
    fil = fil_stradegy(g, fil='node', node_fil='sub', nodefeat = nodefeat)
    g = fil.build_fv()
    for u, v in g.edges():
        assert g[u][v]['fv'] == max(g.node[u]['fv'], g.node[v]['fv'])
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=False)

    # edge feat example
    fil = fil_stradegy(g, fil='edge')
    fil.edgefeat(func='edge_prob', edge_fil='sup')
    g = fil.build_fv()
    for u in g.nodes():
        nbrs = nx.neighbors(g, u)
        nbredgevals = []
        for v in nbrs:
            nbredgevals.append(g[u][v]['fv'])
        assert g.node[u]['fv'] == max(nbredgevals)
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=False)

    # combined fil -- one way
    fil = fil_stradegy(g, fil='combined', nodefeat=nodefeat)
    fil.edgefeat(func='edge_prob', edge_fil='sup')
    g1 = fil.build_fv()
    x = graph2dgm(g1, nodefil='sub', edgefil='sup')
    diagram = x.get_diagram(g1, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=True)
    print(diagram)

    # combined fil -- anathor way. Almost the same with the previous example.
    fil = fil_stradegy(g, fil='combined')
    fil.edgefeat(func='edge_prob', edge_fil='sup')
    fil.nodefeat(nodefeat=nodefeat)
    g2 = fil.build_fv()
    x = graph2dgm(g2, nodefil='sub', edgefil='sup')
    dgm = x.get_diagram(g2, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=True)
    print_dgm(dgm)

