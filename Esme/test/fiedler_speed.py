""" sbm graph classification """

from Esme.dgms.fil import nodefeat
from Esme.graph.function import fil_strategy
from Esme.graph.generativemodel import sbms
from networkx.linalg.algebraicconnectivity import fiedler_vector

if __name__ == '__main__':
    n = 1
    p, q = 0.5, 0.1
    gs = sbms(n=n, n1=100, n2=50, p=p, q=q)
    for i in range(len(gs)):
        g = gs[i]
        # lapfeat = nodefeat(g, 'fiedler', norm=True)
        nodefeat = fiedler_vector(g, normalized=False)  # np.ndarray
        nodefeat = nodefeat.reshape(len(g), 1)

        gs[i] = fil_strategy(g, nodefeat, method='node', viz_flag=False)

    print('Finish computing lapfeat')
