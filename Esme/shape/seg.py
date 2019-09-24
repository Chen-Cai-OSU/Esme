from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import sys

from Esme.dgms.format import flip_dgms
from Esme.dgms.vector import dgms2vec
from Esme.ml.svm import classifier
from Esme.shape.util import face_num, loaddgm, node_num, face_idx, loady
from Esme.ml.eigenpro import eigenpro
from Esme.dgms.fake import permute_dgms
from collections import Counter
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import os.path as osp
from Esme.helper.io_related import make_dir
from Esme.shape.util import load_labels
from Esme.shape.util import prince_cat
from Esme.helper.format import normalize_

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--idx", default=200, type=int,
                    help='model index. Exclude models from 260 to 280')  # (1515, 500) (3026,)

parser.add_argument("--n_iter", default=50, type=int, help='num of iters for eigenpro')  # (1515, 500) (3026,)
parser.add_argument("--clf", default='eigenpro', type=str, help='choose classifier')
parser.add_argument("--test_size", default=0.1, type=float, help='test size')
parser.add_argument("--vec", default='pvector', type=str, help='pl, pvector, ...')
parser.add_argument("--method", default='node_label', type=str, help='node_label or face_label')
parser.add_argument("--seg", default=0, type=int, help='/home/cai.507/Documents/DeepLearning/meshdata/MeshsegBenchmark-1.0/data/seg/Benchmark/31/31_?.seg')


parser.add_argument("--permute", action='store_true')
parser.add_argument("--norm", action='store_true')
parser.add_argument("--new_load", action='store_true', help='load from consistency labels')

def plot_eigpro_res(res1, res2, idx=None, show_flag = False, save_flag = False):

    # res1 = {1: ((0.03345335, 0.6225), (0.0323456, 0.6305732484076433), 0.019212722778320312), 10: ((0.02896553, 0.6645), (0.027736971, 0.6751592356687898), 0.15459346771240234), 16: ((0.027915038, 0.6805), (0.027045922, 0.6751592356687898), 0.21888422966003418), 25: ((0.027005592, 0.7005), (0.026337573, 0.6836518046709129), 0.3047153949737549), 50: ((0.02561936, 0.701), (0.024892189, 0.7091295116772823), 0.5407752990722656)}
    # res2 = {1: ((0.03954246, 0.5725), (0.039126936, 0.5859872611464968), 0.010367870330810547), 10: ((0.03561854, 0.6245), (0.03524615, 0.6326963906581741), 0.09223318099975586), 16: ((0.03479484, 0.639), (0.034788787, 0.6369426751592356), 0.1485884189605713), 25: ((0.03404998, 0.6535), (0.03433756, 0.6369426751592356), 0.22641897201538086), 50: ((0.03294293, 0.662), (0.033193808, 0.6645435244161358), 0.4424324035644531)}

    x, y = [], []
    for k, v in res1.items():
        x.append(k)
        y.append(v[1][1])
    print(x,y)
    plt.scatter(x,y, c='b', marker='o', label='no permute')

    x, y = [], []
    for k, v in res2.items():
        x.append(k)
        y.append(v[1][1])
    print(x, y)
    plt.scatter(x, y, c='r', label='permute')

    plt.legend()

    plt.title(f'perm vs. no permutation for idx {idx}')
    if show_flag: plt.show()
    if save_flag:
        dir = osp.join(osp.dirname(osp.realpath(__file__)),  'Fig', '')
        make_dir(dir)
        file = str(idx) + '.png'
        print(dir + file)
        plt.savefig(dir + file)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    print(prince_cat())
    for k, v in prince_cat().items():
        if args.idx >= k[0] and args.idx <= k[1]:
            print(f'idx {args.idx} is {v}')
            cat = v
            break

    # # seg one shape
    idx = args.idx
    dgms = loaddgm(str(idx), form='dionysus')
    dgms = flip_dgms(dgms)
    if args.permute: dgms = permute_dgms(dgms, permute_flag=True, seed_flag=True)

    # vectorize
    if args.vec == 'pvector':
        dgm_vector = dgms2vec(dgms, vectype='pvector')  # print(np.shape(pd_vector), np.shape(pd_vectors))

    elif args.vec == 'pl':
        kwargs = {'num_landscapes':5, 'resolution':100}
        dgm_vector = dgms2vec(dgms, vectype='pl', **kwargs)

    elif args.vec == 'pi_':
        params = {'bandwidth': 1.0, 'weight': lambda x: x[1], 'im_range': [0, 1, 0, 1], 'resolution': [20, 20]}
        dgm_vector= dgms2vec(dgms, vectype='pi_', **params)
        print(dgm_vector.shape)
        sys.exit()

    elif args.vec == 'pervec':
        kwargs = {'dim':300}
        dgm_vector = dgms2vec(dgms, vectype='pervec')  # print(np.shape(pd_vector), np.shape(pd_vectors))
        dgm_vector = normalize_(dgm_vector)
    else:
        raise Exception(f'No vec like {args.vec}')

    if args.new_load:
        y = load_labels(idx=idx, cat=v)
    else:
        y = loady(model=idx, counter=True, seg=args.seg)

    print(dgm_vector.shape, y.shape)

    X, Y = [],[]
    n_face, n_node = face_num(str(idx)), node_num(str(idx))
    face_x = np.zeros((n_face, dgm_vector.shape[1]))
    face_indices = face_idx(str(idx))
    for i in range(n_face):
        idx1, idx2, idx3 = face_indices[i]
        idx1, idx2, idx3 = int(idx1), int(idx2), int(idx3)
        face_x[i, :] = dgm_vector[idx1][:] + dgm_vector[idx2, :] + dgm_vector[idx3, :]
    print(face_x.shape, y.shape)
    X.append(face_x)
    Y.append(y)

    X, Y = np.concatenate(X), np.concatenate(Y)
    if args.norm: X = normalize(X, axis=0)

    # classifer
    print()
    if args.clf == 'rf':
        clf = classifier(X, Y, method='svm', n_cv=1)
        clf.svm(n_splits=10)
    else:
        kwargs = {}
        res = eigenpro(X, Y, max_iter=args.n_iter, test_size=args.test_size, bd=1, **kwargs)
    print('-' * 150)

    sys.exit()


    # check consistency btwn loady and load_labels
    y = loady(model=args.idx, counter=True, seg=args.seg)
    print(f'old y is of shape {y.shape}')

    y = load_labels(cat='Hand', idx=args.idx)
    print(f'new y is of shape {y.shape}')
    sys.exit()
