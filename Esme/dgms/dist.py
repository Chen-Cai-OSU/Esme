"""
functions related to distance between diagrams
"""

import dionysus as d
import numpy as np
from scipy.spatial.distance import cdist
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.kernel import sw
from Esme.dgms.test import randomdgm
from Esme.helper.format import precision_format

def bd_distance(dgms):
    n = len(dgms)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            m[i][j] = d.bottleneck_distance(dgms[i], dgms[j])
    return m + m.T

def test():
    dgm1 = d.Diagram([(1,2), (2,3)])
    dgm2 = d.Diagram([(1,2), (2,9)])
    dgms = [dgm1, dgm2] * 3
    bd_distance(dgms)

def euc_dist(emb, metric = 'euclidean'):
    # emb = np.random.rand(10,2)
    m = cdist(emb, emb, metric=metric)
    return m

class dgmdist():
    def __init__(self, dgm1, dgm2):
        self.dgm1 = dgm1
        self.dgm2 = dgm2
        self.dgms = [dgm1, dgm2]

    def bd_dist(self):
        return precision_format(d.bottleneck_distance(self.dgm1, self.dgm2))

    def sw_dist(self):
        swdgms = dgms2swdgms(self.dgms)
        res = sw([swdgms[0]], [swdgms[1]], kernel_type='sw', n_directions=10, bandwidth=1.0, K=1, p=1)[0][0]
        sw_dist = np.log(res) * (-2)
        return precision_format(sw_dist)

if __name__ == '__main__':
    dist = dgmdist(randomdgm(2), randomdgm(10))
    dist.sw_dist()
    dist.bd_dist()