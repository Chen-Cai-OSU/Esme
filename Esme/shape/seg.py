from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import sys

from Esme.dgms.format import flip_dgms
from Esme.dgms.vector import dgms2vec
from Esme.ml.svm import classifier
from Esme.shape.util import face_num, loaddgm, node_num, face_idx, loady
from Esme.ml.eigenpro import eigenpro
from Esme.dgms.fake import permute_dgms

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--idx", default=200, type=int, help='model index. Exclude models from 260 to 280') # (1515, 500) (3026,)
parser.add_argument("--n_iter", default=50, type=int, help='num of iters for eigenpro') # (1515, 500) (3026,)
parser.add_argument("--permute", action='store_true')
parser.add_argument("--clf", default='eigenpro', type=str, help='choose classifier')
parser.add_argument("--test_size", default=0.1, type=float, help='test size')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)


    idx=args.idx

    X, Y = [], []
    for idx in  range(1, 6): # [1,2]:
        dgms = loaddgm(str(idx), form='dionysus')
        dgms = flip_dgms(dgms)
        if args.permute:
            dgms = permute_dgms(dgms, permute_flag=True)

        # vectorize
        pd_vector = dgms2vec(dgms, vectype='pvector') # print(np.shape(pd_vector), np.shape(pd_vectors))
        x = pd_vector
        y = loady(idx)
        print(x.shape, y.shape) # (4706, 500) (9408,)

        n_face, n_node = face_num(str(idx)), node_num(str(idx)),

        face_indices = face_idx(str(idx))

        face_x = np.zeros((n_face, x.shape[1]))
        for i in range(n_face):
            idx1, idx2, idx3 = face_indices[i]
            idx1, idx2, idx3 = int(idx1), int(idx2), int(idx3)
            face_x[i,:] = x[idx1][:] + x[idx2,:] + x[idx3,:] # todo check with mathieu on merge stradegy
        print(face_x.shape, y.shape)
        X.append(face_x)
        Y.append(y)

    X, Y = np.concatenate(X), np.concatenate(Y) # todo add normalize
    # from sklearn.preprocessing import normalize
    # X = normalize(X, axis=1)

    # classifer
    if args.clf == 'rf':
        clf = classifier(X, Y, method='svm', n_cv=1)
        clf.svm(n_splits=10)
    else:
        # eigenpro
        eigenpro(X, Y, max_iter=args.n_iter, test_size=args.test_size)
