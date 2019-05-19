import numpy as np
import dionysus as d
def randomdgm(n = 5):
    return d.Diagram(randomdiag(n))

def randomdiag(n = 10):
    res = []
    for _ in range(n):
        res.append([np.random.rand(), np.random.rand()])
    return res

def randomdiags(n = 5):
    res = []
    for _ in range(n):
        res.append(randomdiag(5))
    return res

def randomdgms(n=5):
    res = []
    for _ in range(n):
        res.append(d.Diagram(randomdiag(n)))
    return res

def generate_swdgm(size=100):
    dgm = []
    for i in range(size):
        dgm += [np.random.rand(100,2)]
    return dgm

def dgms2swdgm(dgms):
    swdgms=[]
    for dgm in dgms:
        diag = dgm2diag(dgm)
        swdgms += [np.array(diag)]
    return swdgms

def flip_dgm(dgm):
    import dionysus as d
    for p in dgm:
        if np.float(p.birth) < np.float(p.death):
            return dgm
        assert np.float(p.birth) >= np.float(p.death)
    data = [(np.float(p.death), np.float(p.birth)) for p in dgm]
    return d.Diagram(data)

def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag
