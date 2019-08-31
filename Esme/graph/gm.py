""" similar to 2sbm_nc """

import sys
import numpy as np
import networkx as nx
from networkx.generators.community import stochastic_block_model

# from importlib import reload  # Python 3.4+ only.
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Esme.embedding.lap import LaplacianEigenmaps

from Esme.ml.svm import classifier
from Esme.graph.egograph import egograph
from Esme.dgms.compute import alldgms
from Esme.dgms.stats import dgms_summary
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.kernel import sw_parallel
from Esme.dgms.fake import permute_dgms
from Esme.graph.function import add_edgeval

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n", default=100, help='number of nodes"')
parser.add_argument("--p", default=0.01, type=float, help='The probablity btwn community')
parser.add_argument("--radius", default=1, type=int, help='The radius of the egograph')
parser.add_argument("--rs", default=1, type=int, help='Random seed')
parser.add_argument("--fil", default='lap', type=str, help='Filtration')

if __name__ == '__main__':
    # sys.argv = ['graph/gm.py']
    args = parser.parse_args()
    rs = args.rs
    radius, fil = 1, args.fil
    n_node, p = 100, args.p
    sizes = [n_node] * 3
    permute_flag = True
    labels = [0] * n_node + [1] * n_node + [2] * n_node
    probs = [[0.5, p, p],
             [p, 0.5, p],
             [p, p, 0.5]]

    g = stochastic_block_model(sizes, probs, seed=rs)
    lp = LaplacianEigenmaps(d=1)
    lp.learn_embedding(g, weight='weight')
    lapfeat = lp.get_embedding()
    degfeat = np.array(list(dict(nx.degree(g)).values())).reshape(3 * n_node, 1)
    clf = classifier(degfeat, labels, method=None)
    clf.svm()

    for n in g.nodes():
        g.node[n]['lap'] = float(lapfeat[n,0])
    g = add_edgeval(g, fil=fil)

    ego = egograph(g, radius=radius, n = len(g), recompute_flag=True, norm_flag=True, print_flag=False)
    egographs = ego.egographs(method='serial')
    dgms = alldgms(egographs, radius=radius, dataset='', recompute_flag=True, method='serial', n=n_node)  # compute dgms in parallel


    if permute_flag: dgms = permute_dgms(dgms)
    dgms_summary(dgms)

    swdgms = dgms2swdgms(dgms)
    kwargs = {'bw': 1, 'n_directions': 10}
    sw_kernel, _ = sw_parallel(swdgms, swdgms, kernel_type='sw', parallel_flag=True, **kwargs)
    # sw_distm = np.log()

    clf = classifier(np.zeros((3 * n_node, 10)), labels, method=None, kernel = sw_kernel)
    clf.svm_kernel_()
    sys.exit()



    model = gnn_bl(g, d = 2)
    gnnfeat = model.feat()

    # print(np.dot(gnnfeat.T, gnnfeat).astype(int))
    gnnfeat_distm = cdist(gnnfeat, gnnfeat, metric='euclidean')

    viz_matrix(gnnfeat_distm)
    viz_distm(gnnfeat_distm, y=labels, mode='mds')
    clf = classifier(gnnfeat, labels, method=None)
    clf.svm()

    lp = LaplacianEigenmaps(d=1)
    lp.learn_embedding(g, weight='weight')
    emb = lp.get_embedding()
    lapdistm = cdist(emb, emb, metric='euclidean')
    viz_matrix(lapdistm)

    clf = classifier(emb, labels, method=None)
    clf.svm(n_splits = 2)

    labels_matrix = np.zeros((3 * n, 3))
    for i in range(3*n):
        labels_matrix[i, labels[i]] = 1

    for n_ in g.nodes():
        g.node[n_]['lapfeat'] = emb[n_,:]
        g.node[n_]['gnnfeat'] = gnnfeat[n_, :]
        g.node[n_]['label'] = labels_matrix[n_, :]

    res = diff(g, key='lapfeat', viz_flag=True)
    # res = diff(g, key='gnnfeat', viz_flag=True)
    res = diff(g, key='label', viz_flag=True, dist_type='hamming')

    sys.exit()


