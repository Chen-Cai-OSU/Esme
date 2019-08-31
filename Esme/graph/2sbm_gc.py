""" sbm graph classification """

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import matplotlib.pyplot as plt
import numpy as np

from Esme.dgms.compute import alldgms
from Esme.dgms.format import dgms2swdgms
from Esme.dgms.kernel import sw_parallel
from Esme.embedding.lap import LaplacianEigenmaps
from Esme.graph.function import fil_strategy
from Esme.graph.generativemodel import sbms
from Esme.ml.svm import classifier

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--fil_method", default='edge', type=str, help='number of nodes"')
parser.add_argument("--q", default=0.1, type=float, help='The probablity btwn community')

if __name__ == '__main__':
    # sys.argv = ['graph/2sbm_gc.py']
    args = parser.parse_args()
    print(args)
    n = 100
    p, q = 0.5, args.q
    p_, q_ = 0.4, 0.2
    fil_method = args.fil_method
    zigzag = True if fil_method == 'combined' else False
    edge_kwargs = {'h': 0.3, 'edgefunc': 'edge_prob'}
    gs1 = sbms(n=n, n1=100, n2=50, p=p, q=q)
    gs2 = sbms(n=n, n1=75, n2=75, p=p, q=q)
    # gs3 = sbms(n=n, n1=75, n2=75, p=p_, q=q_)
    # gs3 = sbms(n=n, n1=50, n2=50, p=p, q=q)
    gs = gs2 + gs1
    labels = [1] * n + [2] * n

    plt.title('p: %s, q: %s' % (p, q))
    for i in range(len(gs)):
        g = gs[i]
        lp = LaplacianEigenmaps(d=1)
        lp.learn_embedding(g, weight='weight')
        lapfeat = lp.get_embedding() # lapfeat is an array

        # viz = True if i%100==0 else False
        # plt.subplot(2, 1, 1+i//100)
        # plt.plot(lapfeat)
        # plt.show()

        gs[i] = fil_strategy(g, lapfeat, method=fil_method, viz_flag=False, **edge_kwargs)

    # plt.show()
    # sys.exit()
    # plot node fv value
    # viz fv value
    # val = dict(nx.get_node_attributes(gs[i], 'fv')).values()
    # plt.plot(val)
    # plt.title('q: %s, i: %s'%(q, i))
    # plt.show()
    # sys.exit()

    print('Finish computing lapfeat')
    dgms = alldgms(gs, radius=float('inf'), dataset='', recompute_flag=True, method='serial', n=2 * n, zigzag=zigzag)  # compute dgms in parallel
    print('Finish computing dgms')
    swdgms = dgms2swdgms(dgms)

    feat_kwargs = {'n_directions': 10, 'bw': 1}
    sw_kernel, _ = sw_parallel(swdgms, swdgms, kernel_type='sw', parallel_flag=True, **feat_kwargs)
    clf = classifier(np.zeros((len(labels), 10)), labels, method=None, kernel=sw_kernel)
    print(clf.svm_kernel_())
    print(p, q, edge_kwargs)
