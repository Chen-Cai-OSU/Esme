import sys

from joblib import delayed, Parallel

from Esme.helper.debug import debug
from Esme.helper.load_graph import load_graphs
from Esme.dgms.fil import g2dgm
from Esme.dgms.stats import dgms_summary

def gs2dgms_parallel(n_jobs = 1, **kwargs):
    global gs
    try:
        assert 'gs' in globals().keys()
    except AssertionError:
        print(globals().keys())

    dgms = Parallel(n_jobs=n_jobs)(delayed(g2dgm)(i, g=gs[i], **kwargs) for i in range(len(gs)))
    return dgms

if __name__ == '__main__':
    gs, labels = load_graphs(dataset='imdb_binary')  # step 1
    subdgms = gs2dgms_parallel(n_jobs=1, fil='jaccard', fil_d='sub', one_hom=False, debug_flag=True)  # step2 # TODO: need to add interface
    dgms_summary(subdgms)
    debug(subdgms, 'subdgms')
    sys.exit()
