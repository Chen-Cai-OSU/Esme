import os
import sys

import networkx as nx
import numpy as np
from dgms.stat import dgms_summary
from sklearn.preprocessing import normalize

from Esme.applications.motif.NRL.src.classification import classify, ArgumentParser, ArgumentDefaultsHelpFormatter
from Esme.applications.motif.aux import network2g, nc_prework, load_labels, wadj
from Esme.dgms.compute import alldgms
from Esme.dgms.format import dgms2swdgms
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.egograph import egograph

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--dataset", type=str, default='wikipedia', help='The name of your dataset (used for output)')
parser.add_argument("--n", default=4777, help='number of nodes"')
parser.add_argument("--radius", default=1, type=int, help='The radius of the egograph')

if __name__=='__main__':
    # sys.argv = ['/home/cai.507/.pycharm_helpers/pydev/pydevconsole.py']
    args = parser.parse_args()
    dataset, radius = args.dataset, args.radius
    dir = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/'
    network = os.path.join(dir, 'data/' + dataset + '.mat')
    fil, recomp, norm_flag, n_node = 'deg', True, True, int(args.n)

    mat, A, _, labels_matrix, _, _, indices = load_labels(network, 'network', 'group')
    graph = nx.from_scipy_sparse_matrix(mat['network'])  # TODO: graph rename
    n2g = network2g(dataset=dataset, fil = fil, norm_flag='yes', sub=True, recomp=recomp)

    # spectral clustering
    lp = LaplacianEigenmaps(d = 10)
    lapfeat = lp.learn_embedding(graph, weight='weight')
    lapfeat = normalize(lapfeat, axis=1)
    g = n2g.compute(graph)

    ego = egograph(g, radius=radius, n = n_node, recompute_flag=recomp, norm_flag=norm_flag)
    ego.emb_file(dataset=dataset)
    egographs = ego.egographs(method='batch_parallel')
    dgms = alldgms(egographs, radius=radius, dataset=dataset, recompute_flag=recomp, method = 'serial', n = n_node) # can do parallel computing
    dgms_summary(dgms)

    if True: # sw
        swdgms = dgms2swdgms(dgms)
        kwargs = {'bw':1, 'K':1, 'p':1} #TODO: K and p is dummy here
        # sw_kernel, _ = sw_parallel(swdgms, swdgms, kernel_type='sw', parallel_flag=True, **kwargs)
        # sw_distance = -2 * np.log(sw_kernel)
        # viz_matrix(np.log(sw_distance))
        # viz_distm(sw_distance)

        wg = wadj(g, swdgms=swdgms, n=n_node, **kwargs)
        for u, v in g.edges():
            g[u][v]['lap_weight'] = np.sign(wg[u][v]['weight'])

        lp = LaplacianEigenmaps(d = 100)
        lp.learn_embedding(graph, weight='lap_weight')
        dir = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/EigenPro2/emb/', dataset,'')
        file = 'lap_uwg'
        lp.save_emb(dir, file)

    # classification
    # pdvector = pdemb(dgms, dataset, recompute_flag=True, norm_flag = norm_flag)
    mat_f, dir, file = nc_prework(dataset, norm_flag=True, feat='lap_uwg')
    print(file)
    # file = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/blogcatalog/lap/d_100/emd' # debug
    classify(emb=file + '.npy', network=mat_f, writetofile=True, dataset=dataset,
             algorithm='pdvector', classifier='EigenPro', test_kernel='eigenpro',
             word2vec_format=False, num_shuffles=2, grid_search=False,
             output_dir=dir, training_percents=[0.5, 0.9])

    sys.exit()

    # random gnn
    model = gnn_bl(graph, d=2)

    gnnfeat = model.feat()
    for n in g.nodes():
        g.node[n]['dgm'] = dgms[n]
    # diff(g, key='dgm', dist_type='bd', viz_flag=True)

    for n in graph.nodes():
        graph.node[n]['label'] = np.array(labels_matrix[n,:].todense())
        graph.node[n]['deg'] = nx.degree(graph, n)
        graph.node[n]['gnnfeat'] = gnnfeat[n,:]
        graph.node[n]['lapfeat'] = lapfeat[n,:]

    res = diff(graph, key='lapfeat', viz_flag=True)
    res_ = diff(graph, key='label', dist_type='hamming', viz_flag=True)

    if False: # gnn baseline(random feature)
        model = gnn_bl(graph, d=2)
        gnnfeat = model.feat()
        np.save('/tmp/gnn', gnnfeat)
        mat_f, dir, file = nc_prework(dataset, norm_flag=True, feat='lap_uwg')
        classify(emb= '/tmp/gnn' + '.npy', network=mat_f, writetofile=True, dataset=dataset,
                 algorithm='pdvector', classifier='EigenPro', test_kernel='eigenpro',
                 word2vec_format=False, num_shuffles=1, grid_search=False,
                 output_dir=dir, training_percents=[0.5])
        sys.exit()

