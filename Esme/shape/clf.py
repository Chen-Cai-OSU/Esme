import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter

import dionysus as d
import numpy as np
from joblib import Parallel, delayed

from Esme.dgms.arithmetic import add_dgm
from Esme.dgms.fake import permute_dgms
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.format import load_dgm
from Esme.dgms.kernel import sw_parallel
from Esme.dgms.stats import normalize_
from Esme.dgms.vector import dgms2vec
from Esme.graph.dataset.modelnet import load_modelnet
from Esme.graph.dataset.modelnet import modelnet2graphs
from Esme.ml.eigenpro import eigenpro
from Esme.ml.svm import classifier

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--idx", default=4897, type=int, help='model index. Exclude models from 260 to 280') # (1515, 500) (3026,)
parser.add_argument("--clf", default='eigenpro', type=str, help='choose classifier')
parser.add_argument("--test_size", default=0.1, type=float, help='test size')
parser.add_argument("--n_iter", default=50, type=int, help='num of iters for eigenpro') # (1515, 500) (3026,)
parser.add_argument("--fil", default='cc_w_nbr8_expFalse', type=str, help='filtration')
parser.add_argument("--kernel", default='sw_', type=str, help='kernel type')
parser.add_argument("--version", default='10', type=str, help='10 or 40')


parser.add_argument("--permute", action='store_true')
parser.add_argument("--norm", action='store_true')
parser.add_argument("--random", action='store_true')
parser.add_argument("--ntda", action='store_true')


DIRECT = '/home/cai.507/anaconda3/lib/python3.6/site-packages/save_dgms/' # mn10/fiedler'

def load_clfdgm(idx =1, ntda = False):
    dgm = d.Diagram([[np.random.random(), 1]])
    for fil_d in ['sub']:#['sub', 'sup', 'epd']:
        dir = os.path.join(DIRECT, graph, fil, fil_d, 'norm_True', '')
        if ntda: dir = os.path.join(DIRECT, graph, 'ntda_True',fil, fil_d, 'norm_True', '')
        f = dir + str(idx) + '.csv'

        try:
            tmp_dgm = load_dgm(dir, filename=f)
        except FileNotFoundError:
            print(f'{f} of size {all_dataset[idx].pos.shape[0]}/{all_dataset[idx].face.shape[1]} not found. Added a dummy one')
            tmp_dgm = d.Diagram([[0, 0]])

        dgm = add_dgm(dgm, tmp_dgm)
    # print(f'finsih {idx}-th diagram')
    return dgm


if __name__ == '__main__':
    args = parser.parse_args()

    # check_partial_dgms(DIRECT, graph=graph, fil=fil, fil_d=fil_d, a = 200, b = 1700)
    version = '40'

    train_dataset, test_dataset = load_modelnet(version, point_flag=False)
    if version == '40':  train_dataset = train_dataset[:3632] + train_dataset[3633:3763] + train_dataset[3764:]
    all_dataset = train_dataset + test_dataset
    labels = [int(data.y) for data in all_dataset]

    graph, fil = 'mn' + version, args.fil
    n = len(labels)
    dgms = []

    dgms = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(load_clfdgm)(idx=i, ntda = args.ntda) for i in range(n))
    if args.permute: dgms = permute_dgms(dgms, permute_flag=True)

    if args.kernel == 'sw':
        swdgms = dgms2swdgms(dgms)
        for bw in [0.1,1,10,100]:
            feat_kwargs = {'n_directions': 10, 'bw': bw}
            print(f'star computing kernel...')
            k, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='sw', **feat_kwargs)
            print(k.shape)

            cmargs = {'print_flag': 'off'}  # confusion matrix
            clf = classifier(labels, labels, method='svm', n_cv=1, kernel=k, **cmargs)
            clf.svm_kernel_(n_splits=10)
        sys.exit()

    # convert to vector
    # kwargs = {'num_landscapes': 5, 'resolution': 100, 'keep_zero': True}
    # x = dgms2vec(dgms, vectype='pl', **kwargs)
    kwargs = {'dim':100}
    print('using pervec')
    x = dgms2vec(dgms, vectype='pervec', **kwargs)

    if args.random: x = np.random.random(x.shape)
    if args.norm: x = normalize_(x, axis=0)

    _, y = modelnet2graphs(version=graph[-2:], print_flag=True, labels_only=True)
    print(f'total num is {len(y)}')
    y = np.array(y)
    print(Counter(list(y)))

    # eigenpro
    y = np.array(labels)
    eigenpro(x, y, max_iter=args.n_iter, test_size=args.test_size)
