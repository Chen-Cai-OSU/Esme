" segmentation for multiple shapes "

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter

import numpy as np
from sklearn.preprocessing import normalize

from Esme.dgms.fake import permute_dgms
from Esme.dgms.format import flip_dgms
from Esme.dgms.vector import dgms2vec
from Esme.ml.eigenpro import eigenpro
from Esme.ml.svm import classifier
from Esme.shape.util import face_num, loaddgm, node_num, face_idx, loady, load_labels, get_cat
from Esme.helper.format import precision_format, rm_zerocol, normalize_
from Esme.helper.time import timefunction


def most_frequent(List):
    assert type(List) == list
    if len(List) == 0:
        # raise Exception('List is empty')
        return 0
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

@timefunction
def choose(X, Y, method='node_label', **kwargs):
    """

    :param X: a list of feats
    :param Y: a lisf of labels
    :param method:
    :return:
    """

    # todo some dirty hack
    # kwargs = {'n_face': n_face, 'n_node': n_node, 'face_indices_tuple': face_indices_tuple, 'dgm_vector': dgm_vector, 'y': y}
    # n_face = kwargs.get('n_face', None)
    # n_node = kwargs.get('n_face', None)
    # face_indices_tuple = kwargs.get('face_indices_tuple', None)
    # dgm_vector = kwargs.get('dgm_vector', None)
    # y = kwargs.get('y', None)

    if method == 'node_label':  # mathieu's way
        face_label_dict = dict(zip(face_indices_tuple, y.tolist()))

        pt_labels = []
        for i in range(n_node):
            # filter a dict if i is in the keys
            face_label_dict_ = {k: v for k, v in face_label_dict.items() if i in k}

            # get corresponding labels
            labels = [face_label_dict[k] for k in face_label_dict_]
            lab = most_frequent(labels)
            pt_labels.append(lab)

        X.append(dgm_vector)
        Y.append(np.array(pt_labels))

    elif method == 'face_label':
        face_x = np.zeros((n_face, dgm_vector.shape[1]))
        for i in range(n_face):
            idx1, idx2, idx3 = face_indices[i]
            idx1, idx2, idx3 = int(idx1), int(idx2), int(idx3)
            face_x[i, :] = dgm_vector[idx1][:] + dgm_vector[idx2, :] + dgm_vector[idx3, :]
        print(face_x.shape, y.shape)
        X.append(face_x)
        Y.append(y)

    else:
        raise Exception(f'No such method {method}')

    return X, Y

def get_y(args):
    if args.new_load:
        v = get_cat(idx_)
        y = load_labels(idx=idx_, cat=v)
    else:
        y = loady(model=idx_, counter=True, seg=args.seg)
    return y

@timefunction
def get_vec(dgms, args):
    # vectorize
    if args.vec == 'pvector':
        dgm_vector = dgms2vec(dgms, vectype='pvector')  # print(np.shape(pd_vector), np.shape(pd_vectors))
    elif args.vec == 'pl':
        kwargs = {'num_landscapes': 5, 'resolution': 100, 'keep_zero': True}
        dgm_vector = dgms2vec(dgms, vectype='pl', **kwargs)
        print(f'pl dgmvec is of shape {dgm_vector.shape}')
    elif args.vec == 'pervec':
        kwargs = {'dim': 300}
        dgm_vector = dgms2vec(dgms, vectype='pervec')  # print(np.shape(pd_vector), np.shape(pd_vectors))
        dgm_vector = normalize_(dgm_vector)
    elif args.vec == 'pi_':
        params = {'bandwidth': 0.1, 'im_range': [0, 2, 0, 2], 'resolution': [20, 20]}
        from Esme.dgms.vector import weight_f
        params['weight'] = weight_f(b=3)  # weight function in the paper
        dgm_vector = dgms2vec(dgms, vectype='pi_', **params)
    else:
        raise Exception(f'No vec like {args.vec}')
    return dgm_vector

def clf(X, Y, args):
    # classifer
    if args.clf == 'rf':
        clf = classifier(X, Y, method='svm', n_cv=1)
        clf.svm(n_splits=10)
    else:
        if args.diff_shapes:
            kwargs = {'train_idx': list(range(n_train_dgms))}
        else:
            kwargs = {}
        res = eigenpro(X, Y, max_iter=args.n_iter, test_size=args.test_size, bd=1, **kwargs)
        permute_res.append(res)
        return permute_res
    print('-' * 150)


parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--idx", default=200, type=int,
                    help='model index. Exclude models from 260 to 280')  # (1515, 500) (3026,)

parser.add_argument("--n_iter", default=50, type=int, help='num of iters for eigenpro')  # (1515, 500) (3026,)
parser.add_argument("--clf", default='eigenpro', type=str, help='choose classifier')
parser.add_argument("--test_size", default=0.1, type=float, help='test size')
parser.add_argument("--vec", default='pl', type=str, help='pl, pvector, ...')
parser.add_argument("--method", default='node_label', type=str, help='node_label or face_label')
parser.add_argument("--seg", default=0, type=int, help='/home/cai.507/Documents/DeepLearning/meshdata/MeshsegBenchmark-1.0/data/seg/Benchmark/31/31_?.seg')
parser.add_argument("--n_total_shape", default=2, type=int, help='')
parser.add_argument("--n_train_shape", default=1, type=int, help='')


parser.add_argument("--permute", action='store_true')
parser.add_argument("--norm", action='store_true')
parser.add_argument("--new_load", action='store_true', help='load from consistency labels')
parser.add_argument("--diff_shapes", action='store_true', help='test on different shapes')



if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    idx = args.idx
    n_total_shape = args.n_total_shape
    n_train_shape = args.n_train_shape
    n_test_shape = n_total_shape - n_train_shape
    n_train_dgms = 0
    permute_res = []

    X, Y, n_processed_shape = [], [], 0
    # args.permute = permute

    for idx_ in range(idx, idx + n_total_shape): # [205,207,209,210]: #

        dgms = loaddgm(str(idx_), form='dionysus')
        dgms = flip_dgms(dgms)

        if n_processed_shape < n_train_shape:
            n_train_dgms += len(dgms)
            n_processed_shape += 1

        if args.permute: dgms = permute_dgms(dgms, permute_flag=True, seed_flag=True) # todo change seed_flag back
        dgm_vector = get_vec(dgms, args)
        y = get_y(args)

        print(dgm_vector.shape, y.shape)  # (4706, 500) (9408,)

        n_face, n_node = face_num(str(idx_)), node_num(str(idx_)),
        face_indices = face_idx(str(idx_))  # a list of lists
        face_indices_tuple = [tuple(face) for face in face_indices]  # a list of face tuples

        X, Y = choose(X, Y, method='node_label')

    X, Y = np.concatenate(X), np.concatenate(Y)
    X = rm_zerocol(X)
    if args.norm: X = normalize(X, axis=0)
    print(f'shape of X and Y is {X.shape} and {Y.shape}')
    clf(X, Y, args)


    # assert len(permute_res) == 2 works for permute in [True, False]


