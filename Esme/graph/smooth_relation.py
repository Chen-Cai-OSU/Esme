import sys

import Esme.graphonlib as graphonlib
import networkx as nx
from scipy.spatial.distance import cdist

from Esme.dgms.compute import alldgms
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.egograph import egograph
from Esme.graph.generativemodel import sbm
from Esme.viz.matrix import viz_matrix

if __name__ == '__main__':
    radius, zigzag, fil, n_node = 1, False, 'deg', 100
    # g = nx.random_geometric_graph(n_node,0.3)
    g, labels = sbm(n_node, 0.1)
    lp = LaplacianEigenmaps(d=3)
    lp.learn_embedding(g, weight='weight')
    print(lp.eigval)
    lapfeat = lp.get_embedding()
    lapdist = cdist(lapfeat, lapfeat, metric='euclidean')
    # viz_matrix(lapdist)

    # g = function_basis(g, fil, norm_flag='yes')
    # g = add_edgeval(g, fil)
    a = nx.adjacency_matrix(g).todense()
    p_zhang = graphonlib.smoothing.zhang.smoother(a,h=0.3)  # h : neighborhood size parameter. Example: 0.3 means to include
    for u, v in g.edges():
        g[u][v]['fv'] = p_zhang[u][v]

    for n in g.nodes():
        g.node[n]['fv'] = lapfeat[n,0].astype(float)
    # for u, v in g.edges():
    #     g[u][v]['fv'] = max(g.node[u]['fv'], g.node[v]['fv'])
    ego = egograph(g, radius=radius, n=len(g), recompute_flag=True, norm_flag=True, print_flag=True)
    egographs = ego.egographs(method='parallel')
    dgms = alldgms(egographs, radius=radius, dataset='', recompute_flag=True, method='serial', n=n_node, zigzag=zigzag)  # compute dgms in parallel
    for n in g.nodes():
        g.node[n]['dgm'] = dgms[n]
    sys.exit()

    dist1, dist2, dist3 = [], [], []
    for u, v in g.edges():
        fdist = abs(g.node[u]['fv'] - g.node[v]['fv'])
        bddist = d.bottleneck_distance(g.node[u]['dgm'], g.node[v]['dgm'])
        swdgms = dgms2swdgms([g.node[u]['dgm'], g.node[v]['dgm']])
        swkernel = sw([swdgms[0]], [swdgms[1]], kernel_type='sw', n_directions=10, bandwidth=1.0, K=1, p = 1)[0][0]
        sw_dist = np.log(swkernel) * (-2)

        dist1.append(fdist)
        dist2.append(bddist)
        dist3.append(sw_dist)

    dist1, dist2, dist3 = np.array(dist1), np.array(dist2), np.array(dist3)
    dist1 = np.array(dist1) / np.mean(dist1)
    dist2 = np.array(dist2) / np.mean(dist2)
    dist3 = np.array(dist3) / np.mean(dist3)
    print('Fil function dist is larger than bd dist %s percent of time'%(np.sum(dist1 > dist2) / float(len(dist1))))
    print('bd dist is larger than sw dist %s percent of time' % (np.sum(dist2 > dist3) / float(len(dist1))))
    print('Fil function dist is larger than sw dist %s percent of time' % (np.sum(dist1 > dist3) / float(len(dist1))))

    # plt.plot(dist1, '.', markersize=1, color='b')
    plt.plot(dist2, '.', markersize=1, color='y')
    # plt.plot(dist3, '.', markersize=1, color='r')
    plt.show()
