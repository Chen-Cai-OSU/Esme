import os
import networkx as nx
import numpy as np
import dionysus as d
from Esme.dgms.format import dgms2diags, dgm2diag, diag2dgm, res2dgms, diags2dgms
import networkx as nx
from Esme.dgms.format import diag_check, flip_dgm
from Esme.dgms.stats import print_dgm

class graph2dgm():
    def __init__(self, g):
        self.graph = nx.convert_node_labels_to_integers(g)

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

        f = d.Filtration()
        for simplex, time in simplices:
            f.append(d.Simplex(simplex, time))

        if not zigzag:
            f.sort() if sub else f.sort(reverse=True)
        else:
            f.sort(zigzag_less, reverse=True)
            # print('After zigzag\n')
            # print_f(f)

            # simplices = [([2], 4), ([1, 2], 5), ([0, 2], 6),([0], 1), ([1], 2), ([0, 1], 3)]
            # f = d.Filtration()
            # for vertices, time in simplices:
            #     f.append(d.Simplex(vertices, time))
            # f.append(d.Simplex(vertices, time))
            # f.sort(cmp=zigzag_less,reverse=True)
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
        # only return 0-homology of sublevel filtration TODO: include one homology
        # type can be tuple or pd. tuple can be parallized, pd cannot.
        g = nx.convert_node_labels_to_integers(g)
        simplices = self.get_simplices(g, key = key)
        if one_homology_flag:
            epd_dgm = self.epd(self, g, pd_flag=False)[1]
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

if __name__=='__main__':
    g = nx.random_geometric_graph(1000, 0.1) # create a random graph

    # sublevel filtration. fv stands for function value
    for n in g.nodes():
        g.node[n]['fv'] = np.random.random()
    for u, v in g.edges():
        g[u][v]['fv'] = max(g.node[u]['fv'], g.node[v]['fv'])

    # from filtration on graph nodes to diagram
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag = 'True', one_homology_flag=False, parallel_flag = False, zigzag = False)
    for p in diagram:
        print(p)