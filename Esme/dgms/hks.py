import networkx as nx
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix
from numpy import linalg as LA
import sys
import numpy as np
import matplotlib.pyplot as plt
from Esme.helper.time import timefunction
import time

# @timefunction
def hks(g, t):
    t0 = time.time()
    lap = normalized_laplacian_matrix(g).toarray()
    w, v = LA.eig(lap) # eigvalue, eigvector
    hks = 0
    for i in range(len(w)):
        hks += np.exp(-t * w[i]) * np.multiply(v[i,:], v[i,:])
    hks = hks.reshape(len(v),1)
    if len(g) > 20:
        print(f'hks for graph of size {len(g)} and t {t} takes {time.time()-t0}')
    return hks

if __name__ == '__main__':
    g = nx.random_geometric_graph(2000, 0.1)
    hks_feat = hks(g, 1)
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

