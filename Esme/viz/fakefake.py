import matplotlib
import sys

from Esme.dgms.fake import permute_dgms
from Esme.dgms.fil import gs2dgms_parallel
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.format import print_dgm
from Esme.dgms.kernel import sw_parallel
from Esme.graph.dataset.tu_dataset import load_tugraphs
from Esme.ml.svm import classifier

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn import manifold
from Esme.helper.time import timefunction

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

@timefunction
def diag2m(diag, row_major = True):
    """ from a diagonal matrix to a matrix m[i,j] = m[i,i] """
    # todo: speed up?
    m = np.zeros(diag.shape)
    n = diag.shape[0]
    for i in range(n):
        m[i,:] = diag[i,i]
    if not row_major: m = m.T
    return m

def viz_distm(m, rs = 42, mode='mds', y=None):
    """ Viz a distance matrix using MDS """
    # from importlib import reload  # Python 3.4+ only.

    # points = np.random.random((50, 2))
    # m = cdist(points, points, metric='euclidean')
    #TODO: add legend
    if mode == 'mds':
        mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
        pos = mds.fit_transform(m)
    elif mode == 'tsne':
        tsne = manifold.TSNE(metric='precomputed', verbose=0, random_state=rs)
        pos = tsne.fit_transform(m)
    else:
        raise Exception('No such visualization mode')

    plt.scatter(pos[:, 0], pos[:, 1], c = y, s=2, label='')

    if False:
        n = m.shape[0]//3
        p0, p1, p2 = 0, n, n*2
        plt.scatter(pos[p0:p1, 0], pos[p0:p1, 1], c = 'r', s=2, label='p0')
        plt.scatter(pos[p1:p2, 0], pos[p1:p2, 1], c = 'b', s=2, label='p1')
        plt.scatter(pos[p2:, 0], pos[p2:, 1], c = 'g', s=2, label='p2')
        # plt.legend(loc='upper right')
    plt.legend()
    plt.title('%s viz of matrix of size (%s %s)'%(mode, m.shape[0], m.shape[1]))
    plt.show()

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--graph", default='imdb_binary', type=str, help='graph')
parser.add_argument("--fil", default='deg', type=str, help='graph')
parser.add_argument('--s1', default=42, type=int, help='viz two fake dgms')
parser.add_argument('--s2', default=41, type=int, help='viz two fake dgms')
parser.add_argument('--doublefake', action='store_true', help='viz two fake dgms')
parser.add_argument('--viz', action='store_true', help='viz or not')


if __name__ == '__main__':
    args = parser.parse_args()
    graph = args.graph
    norm = True
    seed_flag = False
    fil = args.fil
    gs, labels = load_tugraphs(graph)
    subdgms = gs2dgms_parallel(gs, fil=fil, fil_d='sub', norm=norm, graph = graph, ntda = False, debug_flag = False)
    supdgms = gs2dgms_parallel(gs, fil=fil, fil_d='sup', norm=norm, graph = graph, ntda = False, debug_flag = False)
    epddgms = gs2dgms_parallel(gs, fil=fil, one_hom=True, norm=norm, graph = graph, ntda = False, debug_flag = False)
    from Esme.helper.parser import combine_dgms
    dgms = combine_dgms(subdgms, supdgms, epddgms, ss=True, epd=False, flip=False)

    true_dgms = dgms
    fake_dgms = permute_dgms(true_dgms, permute_flag=True, seed=args.s1, seed_flag=seed_flag)
    another_fake_dgms = permute_dgms(true_dgms, permute_flag=True, seed=args.s2, seed_flag=seed_flag)
    all_dgms = true_dgms + fake_dgms
    indicator_labels = [1] * len(true_dgms) + [-1] * len(fake_dgms)

    if args.doublefake:
        all_dgms = fake_dgms + another_fake_dgms
        indicator_labels = [1] * len(fake_dgms) + [-1] * len(another_fake_dgms)
    all_dgms = dgms2swdgms(all_dgms)

    # classify true diagrams from fake ones
    feat_kwargs = {'n_directions': 10, 'bw':1}
    k, _ = sw_parallel(all_dgms, all_dgms, parallel_flag=True, kernel_type='sw', **feat_kwargs)
    print(k.shape)
    cmargs = {'print_flag': 'off'} # confusion matrix
    clf = classifier(indicator_labels, indicator_labels, method='svm', n_cv=1, kernel=k, **cmargs)
    clf.svm_kernel_(n_splits=10)
    if not args.viz: sys.exit('-'*50)

    feat_kwargs = {'n_directions': 10, 'bw':1}
    k, _ = sw_parallel(all_dgms, all_dgms, parallel_flag=True, kernel_type='sw', **feat_kwargs)
    print(np.diag(k).shape)
    k_diag = np.diag(np.diag(k))
    kdist = diag2m(k_diag) + diag2m(k_diag, row_major=False) - 2 * k
    assert kdist[1,2] == k[1,1] + k[2,2] -2*k[1,2]
    kdist = np.sqrt(kdist)

    fake_labels = [-label for label in labels]
    viz_distm(kdist, mode='tsne', y= indicator_labels)
    # viz_distm(kdist, mode='tsne', y= labels + fake_labels)

