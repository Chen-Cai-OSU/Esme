import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

import Esme.graphonlib as graphonlib
from Esme.applications.motif.NRL.src.classification import ArgumentParser, ArgumentDefaultsHelpFormatter
from Esme.dgms.compute import alldgms
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.kernel import sw_parallel
from Esme.dgms.stats import dgms_summary
from Esme.dgms.stats import print_dgm
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.egograph import egograph
from Esme.graph.generativemodel import sbm
from Esme.ml.svm import classifier
from Esme.viz.matrix import viz_matrix

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--p", default=0.2, type=float, help='The probablity btwn community')
parser.add_argument("--zigzag", default=False, type=bool, help='Zigzag flag')

if __name__ == '__main__':
    # sys.argv = ['graph/edge_smoothing.py']
    args = parser.parse_args()
    n_node, radius, zigzag = 100, 1, True
    g, labels = sbm(n_node, args.p)
    lp = LaplacianEigenmaps(d=3)
    lp.learn_embedding(g, weight='weight')
    lapfeat = lp.get_embedding()
    lapdistm = cdist(lapfeat, lapfeat, metric='euclidean')
    # viz_matrix(lapdistm)

    a = nx.adjacency_matrix(g).todense()
    p_zhang = graphonlib.smoothing.zhang.smoother(a, h=0.3) # h : neighborhood size parameter. Example: 0.3 means to include
    viz_matrix(p_zhang)

    for n in g.nodes():
        g.node[n]['fv'] = lapfeat[n,0].astype(float)

    # edge based sublevel filtration
    for u, v in g.edges():
        g[u][v]['fv'] = p_zhang[u][v]
    attributes = nx.get_edge_attributes(g, 'fv')
    for n in g.nodes():
        nbrs = nx.neighbors(g, n)
        keys = [key for key in attributes.keys() if n in key]
        vals = [attributes[key] for key in keys]
        g.node[n]['fv'] = max(vals) # min or max?


    if zigzag:
        for u, v in g.edges():
            g[u][v]['fv'] = p_zhang[u][v]
    else:
        for u, v in g.edges():
            g[u][v]['fv'] = max(g.node[u]['fv'], g.node[v]['fv'])

    ego = egograph(g, radius=radius, n=len(g), recompute_flag=True, norm_flag=True, print_flag=False)
    egographs = ego.egographs(method='parallel')
    dgms = alldgms(egographs, radius=radius, dataset='', recompute_flag=True, method='serial', n=n_node, zigzag=zigzag)  # compute dgms in parallel
    dgms_summary(dgms)
    print_dgm(dgms[0])

    swdgms = dgms2swdgms(dgms)
    for bw in [10]:
        kwargs = {'bw': bw, 'K': 1, 'p': 1}  # TODO: K and p is dummy here
        sw_kernel, _ = sw_parallel(swdgms, swdgms, kernel_type='sw', parallel_flag=True, **kwargs)
        sw_distancem = np.log(sw_kernel) * (-2)
        # viz_matrix(sw_distancem)
        clf = classifier(np.zeros((3 * n_node, 10)), labels, method=None, kernel = sw_kernel)
        clf.svm_kernel_()


