"""
functions to maniputate persistence diagrams
"""
import dionysus as d
from Esme.dgms.format import dgm2diag, diag2dgm, array2diag
import numpy as np

def upperdiagonal(dgm):
    for p in dgm:
        assert p.birth <= p.death

def add_dgm(dgm1, dgm2):
    """ add(overlay) two dgms """
    diag1 = dgm2diag(dgm1)
    diag2 = dgm2diag(dgm2)
    data = diag1 + diag2
    if len(data) ==0:
        return d.Diagram([[0,0]])
    return d.Diagram(data)

def add_dgms(dgms1, dgms2):
    """ add two dgms(subdgms and superdgms). """
    dgms = []
    assert len(dgms1) == len(dgms2)
    for i in range(len(dgms1)):
        dgm1, dgm2 = dgms1[i], dgms2[i]
        dgm = add_dgm(dgm1, dgm2)
        dgms.append(dgm)
    return dgms

class dgm_operator(d._dionysus.Diagram):
    def __init__(self, dgms):
        self.dgms = dgms

    def overlay(self):
        diags = []
        for dgm in self.dgms:
            diag = dgm2diag(dgm)
            diags.append(np.array(diag))
        # diags = [np.random.random((3, 2)), np.random.random((2,2))]
        res = np.concatenate(tuple(diags), axis=0) # array
        res = array2diag(res)
        res = diag2dgm(res)
        return res

if __name__=='__main__':
    from Esme.dgms.fake import randomdgms
    dgm1, dgm2 = randomdgms(2)
    add_dgms(dgm1, dgm2)