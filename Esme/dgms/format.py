"""
A few auxliary functions realted to formatting diagrams.
"""
import csv
import os
import time

import dionysus as d
import numpy as np
from sklearn.preprocessing import normalize

def tuple2dgm(tup):
    return d.Diagram(tup)

def diag2array(diag):
    return np.array(diag)

def array2diag(array):
    res = []
    n = len(array)
    for i in range(n):
        p = [array[i,0], array[i,1]]
        res.append(p)
    return res

def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag

def dgmxy(dgm):
    # for plotting
    x, y = [], []
    for p in dgm:
        x.append(p.birth)
        y.append(p.death)
    return x, y

def dgms2diags(dgms):
    t0 = time.time()
    diags = []
    for i in range(len(dgms)):
        diags.append(dgm2diag(dgms[i]))
    print ('Finish converting dgms to diags in %s'%(time.time()-t0))
    return diags

def diag2dgm(diag):
    import dionysus as d
    if type(diag) == list:
      diag = [tuple(i) for i in diag]
    elif type(diag) == np.ndarray:
      diag = [tuple(i) for i in diag] # just help to tell diag might be an array
    dgm = d.Diagram(diag)
    return dgm

def diags2dgms(diags):
    t0 = time.time()
    dgms = []
    for diag in diags:
      dgms.append(diag2dgm(diag))
    print ('Finish converting diags to dgms in %s'%(time.time()-t0))
    return dgms

def diag_check(diag):
    if type(diag)==list and set(map(len, diag)) == {2}:
        return True
    else:
        return False

def res2dgms(res):
    dgms = []
    for diag in res:
        assert diag_check(diag)
        dgms.append(diag2dgm(diag))
    return dgms

def dgms2swdgms(dgms):
    swdgms = []
    for dgm in dgms:
        diag = dgm2diag(dgm)
        swdgms += [np.array(diag)]
    return swdgms

def assert_dgm_above(dgm):
    for p in dgm:
        try:
            assert p.birth <= p.death
        except AssertionError:
            raise Exception('birth is larger than death')

def assert_dgm_below(dgm):
    for p in dgm:
        try:
            assert p.birth >= p.death
        except AssertionError:
            raise Exception('birth is smaller than death')


def assert_dgms_above(dgms):
    for dgm in dgms:
        assert_dgm_above(dgm)

def assert_dgms_below(dgms):
    for dgm in dgms:
        assert_dgm_below(dgm)


def flip_dgm(dgm):
    # flip dgm from below to above, not vise versa
    for p in dgm:
        if np.float(p.birth) < np.float(p.death):
            assert_dgm_above(dgm)
            return dgm
        assert np.float(p.birth) >= np.float(p.death)
    data = [(np.float(p.death), np.float(p.birth)) for p in dgm]
    return d.Diagram(data)

def flip_dgms(dgms):
    dgms_ = []
    for dgm in dgms:
        dgm = flip_dgm(dgm)
        dgms_.append(dgm)
    return dgms_

def normalize_dgm(dgm):
    # dgm = dgms[1]
    diag = np.array(dgm2diag(dgm))
    diag = normalize(diag, axis=0)
    n = len(diag)
    res = []
    for i in range(n):
        res.append(list(diag[i,:]))
    return diag2dgm(res)

def normalize_dgms(dgms):
    res = []
    for dgm in dgms:
        res.append(normalize_dgm(dgm))
    return res

def export_dgm(dgm, dir='./', filename='dgm.csv'):
    # dgm = diagram
    diag = dgm2diag(dgm)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(diag)

def load_dgm(dir='./', filename='dgm.csv'):
    # dir = '/home/cai.507/Documents/DeepLearning/Esmé/dgms/deg/sub/norm_False'
    # filename = '1.csv'
    file = os.path.join(dir, filename)
    with open(file, 'r') as f:
        lines = list(f)
    diag = []
    for line in lines:
        b, d = line.split(',')
        diag.append([float(b), float(d)])
    diag = np.array(diag)
    return diag2dgm(diag)

def normalize_dgm(dgm):
    import numpy as np
    max_ = 0
    for p in dgm:
        max_ = max(max_, max(np.float(abs(p.birth)), np.float(abs(p.death))))
    max_ = np.float(max_)
    data = [(np.float(p.death) / max_, np.float(p.birth) / max_) for p in dgm]
    return d.Diagram(data)

def print_dgm(dgm):
    for p in dgm:
        print(p)



if __name__=='__main__':
    dgm = load_dgm()
    print_dgm(dgm)