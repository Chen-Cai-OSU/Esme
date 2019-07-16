# Replicate permutation test
import sys
import warnings
import argparse

from Esme.dgms.fil import gs2dgms
from Esme.dgms.stats import dgms_summary
from Esme.dgms.arithmetic import add_dgms
from Esme.dgms.vector import dgms2feature, merge_dgms
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

from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np

from pymongo import MongoClient
client = MongoClient('localhost', 27017)
EXPERIMENT_NAME = 'PD'
YOUR_CPU = None  # None is the default setting and will result in using localhost, change if you want something else
DATABASE_NAME = 'tda'
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))

warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    feat_kwargs = {'n_directions': 10, 'granularity':25, 'bw':1}

    # params for classifier
    n_cv = 1
    clf = 'svm'

@ex.main
def main(graph, fil, norm, permute, ss, epd, n_cv, flip, feat, feat_kwargs):
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
    print('kwargs', feat_kwargs)
    label_flag =  dgms_dir_test(fil=fil, fil_d='sub', norm=norm, graph = graph)[1]
    # gs, labels = load_graphs(dataset=graph, labels_only=label_flag)  # step 1
    gs, labels = load_tugraphs(graph, labels_only=True)

    # parallel
    subdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sub', norm=norm, graph = graph)
    supdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sup', norm=norm, graph = graph)
    epddgms = gs2dgms_parallel(n_jobs=-1, fil=fil, one_hom=True, norm=norm, graph = graph)

    dgms = combine_dgms(subdgms, supdgms, epddgms, ss=ss, epd=epd, flip=flip)
    dgms = permute_dgms(dgms, permute_flag=permute)
    dgms_summary(dgms)

    swdgms = dgms2swdgms(dgms)
    if feat == 'sw':
        print(feat_kwargs)
        k, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='sw', **feat_kwargs)
        print(k.shape)
        clf = classifier(labels, labels, method='svm', n_cv=n_cv, kernel=k)
        clf.svm_kernel_(n_splits=10)
        print(clf.stat)
        return clf.stat

    elif feat == 'pi':
        params = {'bandwidth': 1.0, 'weight': (1, 1), 'im_range': [0, 1, 0, 1], 'resolution': [5, 5]}
        images = merge_dgms(subdgms, supdgms, epddgms, vectype='pi', ss=ss, epd=epd, **params)
        clf = classifier(images, labels, method='svm', n_cv=n_cv)
        clf.svm(n_splits=10)
        return clf.stat

    elif feat == 'pss':
        k, _ = sw_parallel(swdgms, swdgms, parallel_flag=False, kernel_type='pss', **feat_kwargs)
        print(k.shape, k, np.max(k))
        clf = classifier(labels, labels, method='svm', n_cv=n_cv, kernel=k)
        clf.svm_kernel_(n_splits=10)
        print(clf.stat)
        return clf.stat

    elif feat == 'wg':
        k, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='wg', **feat_kwargs)
        print(k.shape)
        clf = classifier(labels, labels, method='svm', n_cv=n_cv, kernel=k)
        clf.svm_kernel_(n_splits=10)
        print(clf.stat)
        return clf.stat

    elif feat=='pdvector':
        pass


if __name__ == '__main__':
    ex.run_commandline()  # SACRED: this allows you to run Sacred not only from your terminal,
    sys.exit()

    # load graphs
    gs, labels = load_graphs(dataset=args.graph) # step 1

    # parallel
    subdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sub', norm = norm)
    supdgms = gs2dgms_parallel(n_jobs=-1, fil=fil, fil_d='sup', norm = norm)
    epddgms = gs2dgms_parallel(n_jobs=-1, fil=fil, one_hom=True, norm = norm)

    # serial
    # subdgms = gs2dgms(gs, fil=fil, fil_d='sub', norm=norm, one_hom=False) # step2 # TODO: need to add interface
    # supdgms = gs2dgms(gs, fil=fil, fil_d='sup', norm=norm, one_hom=False)  # step2 #
    # epddgms = gs2dgms(gs, fil=fil, norm=norm, one_hom=True)  # step2 # TODO

    dgms = combine_dgms(subdgms, supdgms, epddgms, args)
    dgms = permute_dgms(dgms, permute_flag=args.permute)
    dgms_summary(dgms)

    # sw kernel
    swdgms = dgms2swdgms(dgms)
    kwargs = {'bw': args.bw, 'n_directions':10, 'K':1, 'p':1}
    sw_kernel, _ = sw_parallel(swdgms, swdgms, parallel_flag=True, kernel_type='sw',  **kwargs)
    print(sw_kernel.shape)

    clf = classifier(labels, labels, method='svm', n_cv=args.n_cv, kernel=sw_kernel)
    clf.svm_kernel_(n_splits=10)
    print(clf.stat)







