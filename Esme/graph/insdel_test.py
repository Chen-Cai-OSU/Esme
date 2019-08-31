# add radnom insertion and deletion and see how PD change

import networkx as nx

from Esme.dgms.compute import graph2dgm
from Esme.dgms.plot import plottwodgms
from Esme.graph.function import function_basis, add_edgeval
from Esme.graph.noise import random_deletion, random_insertion

n_change = 10
g = nx.random_geometric_graph(100, 0.5)
g = function_basis(g, 'random')
g = add_edgeval(g, fil='random')

m0 = graph2dgm(g)
dgm = m0.get_diagram(g)

# deletion
g = random_deletion(g, n = n_change, print_flag = True)
m1 = graph2dgm(g)
dgm_del = m1.get_diagram(g)

# insertion
g = random_insertion(g, n = n_change, print_flag = True)
g = add_edgeval(g, fil='random')
m2 = graph2dgm(g)
dgm_ins = m2.get_diagram(g)

# print_dgm(dgm)
# print_dgm(dgm_after)
plottwodgms(dgm, dgm_ins)
