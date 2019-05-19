from Esme.dgms.format import dgm2diag
import numpy as np
import matplotlib.pyplot as plt
from Esme.dgms.test import randomdgm
from Esme.dgms.dist import dgmdist

def plottwodgms(dgm1, dgm2):
    diag1 = np.array(dgm2diag(dgm1))
    diag2 = np.array(dgm2diag(dgm2))
    plt.scatter(diag1[:, 0], diag1[:, 1], alpha=0.3)
    plt.scatter(diag2[:, 0], diag2[:, 1], alpha=0.3)
    m = dgmdist(dgm1, dgm2)
    plt.title('bd is %s, sw is %s'%(m.bd_dist(), m.sw_dist()))
    plt.show()

if __name__ == '__main__':
    dgm1 = randomdgm(10)
    dgm2 = randomdgm(5)
    plottwodgms(dgm1, dgm2)