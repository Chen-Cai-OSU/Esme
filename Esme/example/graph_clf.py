import numpy as np

from Esme.applications.motif.NRL.src.classification import ArgumentParser, ArgumentDefaultsHelpFormatter
from Esme.dgms.fil import gs2dgms
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.kernel import sw_parallel
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.function import fil_strategy
from Esme.graph.generativemodel import sbms
from Esme.ml.svm import classifier

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--fil_method", default='edge', type=str, help='number of nodes"')

if __name__ == '__main__':
    # set parameters
    args = parser.parse_args()
    n = 100
    p, q = 0.5, 0.1
    p_, q_ = 0.4, 0.2
    fil_method = args.fil_method
    zigzag = True if fil_method == 'combined' else False

    # generate data
    gs1 = sbms(n=n, n1=100, n2=50, p=p, q=q)
    gs2 = sbms(n=n, n1=75, n2=75, p=p, q=q)
    gs = gs2 + gs1
    labels = [1] * n + [2] * n

    # node filtration is fiedler vector.
    # edge_kwargs = {'h': 0.3, 'edgefunc': 'edge_prob'}
    # for i in range(len(gs)):
    #     g = gs[i]
    #     lp = LaplacianEigenmaps(d=1)
    #     lp.learn_embedding(g, weight='weight')
    #     lapfeat = lp.get_embedding()
    #     gs[i] = fil_strategy(g, lapfeat, method=fil_method, viz_flag=False, **edge_kwargs)
    # print('Finish computing lapfeat')

    # compute diagrams
    dgms = gs2dgms(gs, fil='deg', fil_d='sub', norm=True)

    # compute kernel and evaluation
    swdgms = dgms2swdgms(dgms)
    kwargs = {'bw': 1, 'n_directions': 10}
    sw_kernel, _ = sw_parallel(swdgms, swdgms, kernel_type='sw', parallel_flag=False, **kwargs)
    clf = classifier(np.zeros((len(labels), 10)), labels, method=None, kernel=sw_kernel)
    print(clf.svm_kernel_())
