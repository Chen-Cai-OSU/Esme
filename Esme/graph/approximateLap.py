from __future__ import division
from __future__ import print_function

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from importlib import reload  # Python 3.4+ only.
from Esme.applications.motif.NRL.src.classification import ArgumentParser, ArgumentDefaultsHelpFormatter
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.generativeModel import sbm
from helper.rs import *
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from Esme.viz.graph import viz_graph

from GN.gcn.gcn.models import GCN, MLP
from GN.gcn.gcn.random_permutation import random_features
from GN.gcn.gcn.utils import *

np.random.seed(42)

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--p", default=0.3, type=float, help='The probablity btwn community')

def load_lapdata(g, d=10, version='regression'):
    # g = nx.random_geometric_graph(100, 0.1)
    lp = LaplacianEigenmaps(d=d)
    lp.learn_embedding(g, weight='weight')
    emb = lp.get_embedding()
    # https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
    labels = np.zeros_like(emb)
    labels[np.arange(len(emb)), emb.argmax(1)] = 1
    labels = labels.astype(int)
    assert np.sum(labels) == emb.shape[0]
    # labels = (abs(emb) == abs(emb).max(axis=1)[:, None]).astype(int)

    adj = nx.adjacency_matrix(g)
    if version == 'regression':
        labels = emb
        features = np.random.random((len(g), d))  # featureless

    features = emb #np.random.random(emb.shape)#emb #TODO: change back
    features = lil_matrix(features)

    mask = np.zeros((3, len(g)))
    idx = np.random.choice([0, 1, 2], len(g), p=[0.5, 0.2, 0.3])
    for i in range(len(g)):
        mask[idx[i], i] = 1
    mask = mask.astype(bool)
    train_mask, val_mask, test_mask = mask[0], mask[1], mask[2]
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.') # change from 0.01 to 0.005
flags.DEFINE_integer('epochs', 20000, 'Number of epochs to train.') # change from 200 t0 2000
flags.DEFINE_integer('hidden1', 160, 'Number of units in hidden layer 1.') # change from 16 to 160
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).') # change from 0.5 to 0
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 20000, 'Tolerance for early stopping (# of epochs).') # change from 10 to 1000
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

if __name__ == '__main__':
    # sys.argv = ['graph/gm.py']
    args = parser.parse_args()
    d = 2
    g, labels = sbm(n = 200, p =args.p)

    # Load data
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_lapdata(g, d=d)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_lapdata(g, d=d, version='regression')

    if False: # for debug
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
        g = nx.from_scipy_sparse_matrix(adj)
        lp = LaplacianEigenmaps(d=6)
        lp.learn_embedding(g, weight='weight')
        emb = lp.get_embedding()
        # b = np.zeros_like(emb)
        # b[np.arange(len(emb)), emb.argmax(1)] = 1
        # b = b.astype(int)
        eigen_labels = (abs(emb) == abs(emb).max(axis=1)[:, None]).astype(int)  # https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
        y_train = random_labels(y_train)
        assert eigen_labels.shape == y_train.shape
        n_row = y_train.shape[0]
        index_array = np.sum(y_train, axis=1).reshape(n_row, 1)
        y_train = np.multiply(eigen_labels, index_array)

    features = random_features(features)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())
    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        if epoch % 50 ==0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    # sys.exit()


    n = 100
    g, labels = sbm(100, 0.01)
    lp = LaplacianEigenmaps(d=10)
    lp.learn_embedding(g, weight='weight')
    emb = lp.get_embedding()

    train_indices, test_indices = train_test_split(range(n*3), shuffle=True)
    for n in g.nodes():
        g.node[n]['lap'] = emb[n, :].astype('float')
        g.node[n]['train'] = True if n in train_indices else False
    colors = np.vstack(nx.get_node_attributes(g, 'lap').values())
    colors = (abs(colors) == abs(colors).max(axis=1)[:, None]).astype(int)

    plt.figure(1)
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        show_flag = True if i==9 else False
        viz_graph(g, node_size=5, edge_width=0.01, node_color=colors[:, i-1], show=show_flag)



