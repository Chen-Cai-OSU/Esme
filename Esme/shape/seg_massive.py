""" quick test file """

# Replicate permutation test
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

from Esme.dgms.fake import permute_dgms
from Esme.dgms.fil import g2dgm
from Esme.dgms.format import flip_dgms
from Esme.dgms.ioio import dgms_dir_test, load_dgms, save_dgms
from Esme.dgms.vector import dgms2vec
from Esme.helper.format import normalize_
from Esme.ml.eigenpro import eigenpro
from Esme.ml.svm import classifier
from Esme.db.util import db_config
from Esme.shape.util import face_num, loaddgm, node_num, face_idx, loady, prince_cat


warnings.filterwarnings("ignore", category=DeprecationWarning)
ex = db_config(exp_name = 'Shape', database_name = 'tda_shape')

@ex.capture
def gs2dgms_parallel(n_jobs = 1, **kwargs):
    """ a wraaper of g2dgm for parallel computation """

    if dgms_dir_test(**kwargs)[1]:
        dgms = load_dgms(**kwargs)
        return dgms
    try:
        assert 'gs' in globals().keys()
    except AssertionError:
        print(globals().keys())
    dgms = Parallel(n_jobs=n_jobs)(delayed(g2dgm)(i, gs[i], **kwargs) for i in range(len(gs)))
    save_dgms(dgms, **kwargs)
    return dgms


@ex.config
def get_config():
    # params for data
    idx = 200
    n_iter = 50
    clf = 'eigenpro'
    test_size = 0.1
    vec = 'pl'
    method = 'node_label'
    seg = 0

    permute = False
    norm = False

@ex.main
def main(idx, n_iter, clf, test_size, vec, method, seg, permute, norm):

    cat_dict = prince_cat()
    for k, v in cat_dict.items():
        if idx >= k[0] and idx <= k[1]:
            print(f'idx {idx} is {v}')
            break

    # seg one shape
    dgms = loaddgm(str(idx), form='dionysus')
    dgms = flip_dgms(dgms)
    if permute: dgms = permute_dgms(dgms, permute_flag=True, seed_flag=True)

    # vectorize
    if vec == 'pvector':
        dgm_vector = dgms2vec(dgms, vectype='pvector')  # print(np.shape(pd_vector), np.shape(pd_vectors))
    elif vec == 'pl':
        kwargs = {'num_landscapes': 5, 'resolution': 100}
        dgm_vector = dgms2vec(dgms, vectype='pl', **kwargs)
    elif vec == 'pervec':
        kwargs = {'dim': 300}
        dgm_vector = dgms2vec(dgms, vectype='pervec', **kwargs)  # print(np.shape(pd_vector), np.shape(pd_vectors))
        dgm_vector = normalize_(dgm_vector)
    else:
        raise Exception(f'No vec like {vec}')

    y = loady(model=idx, counter=True, seg=seg)

    X, Y = [], []
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
    if norm: X = normalize(X, axis=0)
    print(f'X is of shape {dgm_vector.shape} and Y is of shape {y.shape}\n')

    # classifer
    if clf == 'rf':
        clf = classifier(X, Y, method='svm', n_cv=1)
        res = clf.svm(n_splits=10) # todo res format
    else:
        kwargs = {}
        res = eigenpro(X, Y, max_iter=n_iter, test_size=test_size, bd=1, **kwargs)
    print('-' * 150)

    return res



if __name__ == '__main__':
    ex.run_commandline()  # SACRED: this allows you to run Sacred not only from your terminal,







