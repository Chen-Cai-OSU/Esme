import numpy
import numpy as np
import scipy as sp
import sklearn.linear_model as skl
import sklearn.metrics
from link_prediction_code import *
from six import iteritems
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as skshuffle
import scipy.io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
parser = ArgumentParser()
parser.add_argument('-input_graph', type=str, default="example_graphs/ppi_lp.mat")
parser.add_argument('-input_embedding0', type=str, default="example_graphs/ppi.embeddings_u.npy")
parser.add_argument('-input_embedding1', type=str, default="example_graphs/ppi.embeddings_v.npy")
parser.add_argument('-share_embeddings', type=bool, default=True)
args = parser.parse_args()

u = np.load(args.input_embedding0)
if(args.share_embeddings):
    v = u
else:
    v = np.load(args.input_embedding1)
graph = sp.io.loadmat(args.input_graph)
train_adj = graph['network']
test_adj = graph['network_te']
adj = ((train_adj+test_adj)>0).astype(int)

G = nx.from_scipy_sparse_matrix(adj).to_directed()
G_train = nx.from_scipy_sparse_matrix(train_adj).to_directed()
G_test = nx.from_scipy_sparse_matrix(test_adj).to_directed()

print("aaa",test_adj.shape,u.shape,v.shape)
pos_tr = graph['pos_tr']
pos_te = graph['pos_te']
neg_tr = graph['neg_tr']
neg_te = graph['neg_te']

for i in [None,'hadamard','average','l1','l2']:
    print(compute_auc_score(pos_tr,neg_tr,pos_te,neg_te,u,v,i))
##for i in [2,10,100,200,300,500,800,1000,10000]:
##    print(compute_link_prediction(u,v,G,G_train,G_test,metric='pr@k',k=i))
#print(compute_link_prediction(u,v,G,G_train,G_test,metric='MAP'))
