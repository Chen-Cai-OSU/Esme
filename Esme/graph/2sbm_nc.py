"""
classify nodes of a graph sampled from stochastic block model
compute the persistence diagram based on node value (fiedler vector) and edge value (edge probability)
use sliced wasserstein kernel to classify diagrams
"""

import numpy as np
from scipy.spatial.distance import cdist

from Esme.dgms.compute import alldgms
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.kernel import sw_parallel
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.egograph import egograph
from Esme.graph.function import fil_strategy
from Esme.graph.generativemodel import sbm2
from Esme.ml.svm import classifier

if __name__ == '__main__':
    radius, zigzag, fil, n1, n2 = 1, True, 'deg', 150, 150
    fil_method = 'combined'
    g, labels = sbm2(n1=n1, n2=n2, p=0.5, q=0.2)

    lp = LaplacianEigenmaps(d=1)
    lp.learn_embedding(g, weight='weight')
    lapfeat = lp.get_embedding()
    lapdist = cdist(lapfeat, lapfeat, metric='euclidean')

    kwargs = {'h': 0.3}
    g = fil_strategy(g, lapfeat, method=fil_method, viz_flag=False, **kwargs)

    ego = egograph(g, radius=radius, n=len(g), recompute_flag=True, norm_flag=True, print_flag=False)
    egographs = ego.egographs(method='serial')
    dgms = alldgms(egographs, radius=radius, dataset='', recompute_flag=True, method='serial', n=n1+n2, zigzag=zigzag)  # compute dgms in parallel

    swdgms = dgms2swdgms(dgms)
    kwargs = {'bw': 1, 'n_directions':10}
    sw_kernel, _ = sw_parallel(swdgms, swdgms, kernel_type='sw', parallel_flag=True, **kwargs)
    clf = classifier(np.zeros((n1+n2, 10)), labels, method=None, kernel=sw_kernel)
    clf.svm_kernel_()

