import dionysus as d
import numpy as np
import time
from Esme.helper.format import precision_format as pf
import numpy as np
import ripser
from ripser import ripser, plot_dgms
from ripser import Rips

def test_rips(n = 100, dim = 2):
    # dionysus version
    t0 = time.time()
    points = np.random.random((n,dim))
    f = d.fill_rips(points, 2, 0.3)
    print(f)
    # for s in f:
    #     print(s)
    print('Computing rips of %s nodes at dim %s takes %s'%(n, dim, pf(time.time()-t0)))

def load_emb():
    direct = '/home/cai.507/Documents/DeepLearning/gae/experiments/ppi/1/100/320/64/'
    emb = np.load(direct + 'emb_nc.npy')
    return emb

def test_fastrips(n = 100, dim = 2):
    t0 = time.time()
    rips = Rips()
    data = np.random.random((n, dim))
    diagrams = rips.fit_transform(data)
    rips.plot(diagrams, title='n = %s, d = %s'%(n,d))
    print('Computing rips of %s nodes at dim %s takes %s' % (n, dim, pf(time.time() - t0)))

def pdofemb():
    rips = Rips()
    data = load_emb()
    diagrams = rips.fit_transform(data)
    rips.plot(diagrams, title='n = %s, d = %s' % (n, d))


for n in [100, 1000, 10000]:
    test_fastrips(n, dim=256)
