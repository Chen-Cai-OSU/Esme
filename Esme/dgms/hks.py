import networkx as nx
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix
from numpy import linalg as LA
import sys
import numpy as np
import matplotlib.pyplot as plt
from Esme.helper.time import timefunction
from Esme.helper.format import precision_format
from Esme.dgms.fil import nodefeat
import time

@timefunction
def hks(g, t):
    t0 = time.time()
    lap = normalized_laplacian_matrix(g, weight='weight').toarray()
    w, v = LA.eig(lap) # eigvalue, eigvector
    hks = 0
    for i in range(len(w)):
        hks += np.exp(-t * w[i]) * np.multiply(v[i,:], v[i,:])
    hks = hks.reshape(len(v),1)
    if len(g) > 20 and time.time()-t0 > 5:
        print(f'hks for graph of nodes/edges {len(g)}/{len(g.edges())} and t {t} takes {precision_format(time.time()-t0, 2)}')
    return hks

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--size", default=1000, type=int, help='The size of graph') # (1515, 500) (3026,)

if __name__ == '__main__':
    args = parser.parse_args()
    # g = nx.random_geometric_graph(args.size, 0.01)
    g = nx.random_tree(n = args.size)
    print(nx.info(g))
    hks_feat = hks(g, 1)

    t0 = time.time()
    feat = nodefeat(g, 'fiedler')
    print(f'fielder takes {time.time() - t0}')

    print(hks_feat.shape)
    sys.exit()

    # ignore
    lap = normalized_laplacian_matrix(g).toarray()
    assert (lap == lap.T).all()
    w, v = LA.eig(lap) # eigvalue, eigvector
    print(w.shape, v.shape)

    t = 1
    hks = 0
    for i in range(len(w)):
        hks += np.exp(-t * w[i]) * np.multiply(v[i,:], v[i,:])
    print(hks.shape)
    sys.exit()
    # check if computation makes sense
    dotlist = []
    for i in range(100):
        tmp = np.abs(np.dot(v[1,:], v[i,:]))
        dotlist.append(tmp)
    plt.plot(dotlist)
    plt.show()

