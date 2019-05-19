import sys, os
import numpy as np
import networkx as nx
import warnings
from embedding.lap import LaplacianEigenmaps
from scipy.spatial.distance import cdist
from viz.matrix import viz_eigen, viz_matrix
import time

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from NodeClassification.NRL.classification import load_embeddings, ArgumentParser, ArgumentDefaultsHelpFormatter, classify
from applications.motif.aux import load_labels

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--network", type=str, help='The path and name of the .mat file containing the adjacency matrix and node labels of the input network')
parser.add_argument("--dataset", type=str, default='wikipedia', help='The name of your dataset (used for output)')

# sys.argv = ['/home/cai.507/.pycharm_helpers/pydev/pydevconsole.py']
args = parser.parse_args()
dataset = args.dataset
grid_search, word2vec_format, embedding_params = False, False, None

direct = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/'
emb = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/'+ dataset + '/lap/d_100/emd.npy'

# step 1
network = os.path.join(direct, 'data/' + dataset + '.mat')
mat, A, graph, labels_matrix, labels_count, mlb, indices = load_labels(network, 'network', 'group')
graph = nx.from_scipy_sparse_matrix(mat['network'])
# for u,v in graph.edges():
#     graph[u][v]['jac'] = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
# emb = '/tmp/test_feat.npy'
# feat = np.load(emb)
LE = LaplacianEigenmaps(d=100)
# feat = LE.learn_embedding(graph, weight='weight')
# feat = feat.astype(float)
from sklearn.utils.extmath import randomized_svd
# L_sym = nx.normalized_laplacian_matrix(graph, weight='weight')
# t0 = time.time()
# u, v, w = randomized_svd(L_sym, n_components=500, n_iter=30)
# feat_fast = (u + w.T)* 0.5
# feat = np.zeros(feat_fast.shape)
# print('Takes %s using randomized svd'%(time.time()-t0))
# np.save('/tmp/svd_feat', feat_fast)
# emb = '/tmp/svd_feat.npy'
# viz_matrix((feat - feat_fast))

features_matrix, normalized_features_matrix = load_embeddings(emb, word2vec_format, graph, indices)


if False:
    viz_matrix(kernel)
    viz_eigen(kernel, start=1, end=500)


# cdist(features_matrix, features_matrix, metric = 'euclidean')
# rk = np.rank(distm)
# print('Finish computing distance matrix, rank is %s'%rk)

from viz.matrix import viz_distm, viz_matrix
# viz_matrix(distm)

# step 3
training_percents = [0.1, 0.5, 0.9]
mat_f = os.path.join('/home/cai.507/Documents/DeepLearning/EmbeddingEval/data/',  dataset + '.mat')

classify(emb=emb, network=mat_f, writetofile=False, classifier = 'EigenPro', test_kernel="eigenpro",
                                                dataset=dataset, algorithm='diffkernel', word2vec_format=False,
                                                num_shuffles=1, output_dir='/tmp/',
                                                training_percents=training_percents)
#
# classify(emb=emb, network=mat_f, writetofile=False, dataset=dataset + '/mat',
#                                             algorithm='diffkernel', word2vec_format=False, num_shuffles=2, classifier='LR', grid_search=True,
#                                             output_dir='/tmp/', training_percents = training_percents)



