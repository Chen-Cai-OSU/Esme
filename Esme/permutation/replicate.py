# Replicate permutation test
import sys

from Esme.dgms.fil import gs2dgms
from Esme.dgms.stats import dgms_summary
from Esme.dgms.arithmetic import add_dgms
from Esme.dgms.vector import dgms2vec, merge_dgms
from Esme.helper.load_graph import load_graphs
from Esme.ml.svm import classifier
from Esme.dgms.fake import permute_dgms
from Esme.dgms.kernel import sw, sw_parallel
from Esme.dgms.format import dgms2swdgms
from Esme.helper.parser import combine_dgms
from Esme.helper.debug import debug
from joblib import Parallel, delayed
from Esme.dgms.fil import g2dgm
from Esme.dgms.ioio import dgms_dir_test, load_dgms, save_dgms
from Esme.graph.dataset.tu_dataset import load_tugraphs
from Esme.graph.dataset.tu_dataset import graphs_stat
from Esme.permutation.mongo_util import set_ex
from Esme.helper.format import print_line
from Esme.permutation.mongo_util import check_duplicate, get_tda_db

import numpy as np

from pymongo import MongoClient
ex = set_ex()

@ex.capture
def gs2dgms_parallel(n_jobs = 1, **kwargs):
    """ a wraaper of g2dgm for parallel computation
        sync with the same-named function in fil.py
        put here for parallelizaiton reason
    """

    if dgms_dir_test(**kwargs)[1]: #and kwargs.get('ntda', None)!=True: # load only when ntda=False
        dgms = load_dgms(**kwargs)
        return dgms
    try:
        assert 'gs' in globals().keys()
    except AssertionError:
        print(globals().keys())

    try:
        # print('in gs2dgms_parallel', kwargs)
        dgms = Parallel(n_jobs=n_jobs)(delayed(g2dgm)(i, gs[i], **kwargs) for i in range(len(gs)))
    except NameError:  # name gs is not defined
        sys.exit('NameError and exit')

    save_dgms(dgms, **kwargs)
    return dgms

@ex.config
def get_config():
    # params for data
    fil = 'ricci'
    graph = 'mutag'
    norm = True
    permute = False

    # params for diagram
    ss = True
    epd = False
    flip = False

    # params for featuralization
    feat = 'sw'
    feat_kwargs = {} #{'n_directions': 10, 'granularity':25, 'bw':1}

    # params for classifier
    n_cv = 1
    clf = 'svm'

    # params for ntda (if True. Don't even use PD)
    ntda = False


@ex.main
def main(graph, fil, norm, permute, ss, epd, n_cv, flip, feat, feat_kwargs, ntda):
    """
    All hyperprameter goes here.

    :param graph: graph dataset
    :param fil: filtration function
    :param norm: normalize or not
    :param permute: whether permute dgm
    :param ss: both sublevel and superlevel or not
    :param epd: include extended persistence or not
    :param n_cv: number of cross validation
    :return:
    """

    global gs
    print('feat kwargs', feat_kwargs)
    db = get_tda_db()
    params = {'graph':graph, 'fil': fil, 'norm': norm, 'permute': permute, 'ss': ss, 'epd': epd,
              'n_cv': n_cv, 'flip': flip, 'feat': feat, 'ntda':ntda, 'feat_kwargs': feat_kwargs}
    if check_duplicate(db, params): return

    label_flag =  dgms_dir_test(fil=fil, fil_d='sub', norm=norm, graph = graph)[1]
    # gs, labels = load_graphs(dataset=graph, labels_only=label_flag)  # step 1
    gs, labels = load_tugraphs(graph, labels_only=False) # labels_only true means gs is None. Turned on for high speed

    # parallel

    # subdgms = gs2dgms(gs, n_jobs=-1, fil=fil, fil_d='sub', norm=norm, graph = graph, ntda = ntda, debug_flag=True)
    subdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sub', norm=norm, graph = graph, ntda = ntda)
    supdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sup', norm=norm, graph = graph, ntda = ntda)
    epddgms = gs2dgms_parallel(n_jobs=-1, fil=fil, one_hom=True, norm=norm, graph = graph, ntda = ntda)

    dgms = combine_dgms(subdgms, supdgms, epddgms, ss=ss, epd=epd, flip=flip)
    dgms = permute_dgms(dgms, permute_flag=permute) # old way
    dgms_summary(dgms)

    swdgms = dgms2swdgms(dgms)
    if feat == 'sw':
        print(feat_kwargs)
        k, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='sw', **feat_kwargs)
        print(k.shape)
        cmargs = {'print_flag': 'off'} # confusion matrix
        clf = classifier(labels, labels, method='svm', n_cv=n_cv, kernel=k, **cmargs)
        clf.svm_kernel_(n_splits=10)

    elif feat == 'pi': # vector
        params = {'bandwidth': 1.0, 'weight': (1, 1), 'im_range': [0, 1, 0, 1], 'resolution': [5, 5]}
        images = merge_dgms(subdgms, supdgms, epddgms, vectype='pi', ss=ss, epd=epd, **params)
        clf = classifier(images, labels, method='svm', n_cv=n_cv)
        clf.svm(n_splits=10)

    elif feat == 'pss':
        k, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='pss', **feat_kwargs)
        # print(k.shape, k, np.max(k))
        clf = classifier(labels, labels, method='svm', n_cv=n_cv, kernel=k)
        clf.svm_kernel_(n_splits=10)

    elif feat == 'wg':
        k, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='wg', **feat_kwargs)
        print(k.shape)
        clf = classifier(labels, labels, method='svm', n_cv=n_cv, kernel=k)
        clf.svm_kernel_(n_splits=10)

    elif feat=='pervec':
        cmargs = {'print_flag': 'on'} # confusion matrix
        pd_vector = dgms2vec(dgms, vectype='pervec', **feat_kwargs)
        clf = classifier(pd_vector, labels, method='svm', n_cv=n_cv, **cmargs)
        clf.svm(n_splits=10)

    elif feat == 'pf':
        k, _ = sw_parallel(swdgms, swdgms, parallel_flag=False, kernel_type='pf', **feat_kwargs)
        clf = classifier(labels, labels, method='svm', n_cv=n_cv, kernel=k)
        clf.svm_kernel_(n_splits=10)
    else:
        raise Exception('No such feat %s'%feat)

    print(clf.stat)
    print_line()
    return clf.stat

def get_params():
    params = get_config()
    # params.pop('feat_kwargs', None) # not doing exact matching here
    print('default params for sacred: ', params)  # {'fil': 'ricci', 'graph': 'mutag', 'norm': True, 'permute': False, 'ss': True, 'epd': False, 'flip': False, 'feat': 'sw', 'feat_kwargs': {}, 'n_cv': 1, 'clf': 'svm'}

    return params

if __name__ == '__main__':
    db = get_tda_db()
    params = get_params()
    if not check_duplicate(db, params):
        ex.run_commandline()  # SACRED: this allows you to run Sacred not only from your terminal,
    sys.exit()

    # load graphs
    gs, labels = load_graphs(dataset=args.graph)

    # parallel
    subdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sub', norm = norm)
    supdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sup', norm = norm)
    epddgms = gs2dgms_parallel(n_jobs=-1, fil=fil, one_hom=True, norm = norm)

    # serial
    # subdgms = gs2dgms(gs, fil=fil, fil_d='sub', norm=norm, one_hom=False) # step2 # TODO: need to add interface
    # supdgms = gs2dgms(gs, fil=fil, fil_d='sup', norm=norm, one_hom=False)  # step2 #
    # epddgms = gs2dgms(gs, fil=fil, norm=norm, one_hom=True)  # step2 # TODO

    dgms = combine_dgms(subdgms, supdgms, epddgms, args)
    dgms = permute_dgms(dgms, permute_flag=args.permute, permute_ratio = 0.5)
    dgms_summary(dgms)

    # sw kernel
    swdgms = dgms2swdgms(dgms)
    kwargs = {'bw': args.bw, 'n_directions':10, 'K':1, 'p':1}
    sw_kernel, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='sw',  **kwargs)
    print(sw_kernel.shape)

    clf = classifier(labels, labels, method='svm', n_cv=args.n_cv, kernel=sw_kernel)
    clf.svm_kernel_(n_splits=10)
    print(clf.stat)







