"""
functions related to generate fake diagrams from true diagrams
"""

from Esme.dgms.format import dgm2diag, diag2dgm
from Esme.dgms.test import randomdgms
from Esme.dgms.stats import print_dgm, dgm_filter
import networkx as nx
from numpy import random
import dionysus as d
import sys
import numpy as np
import random as random_
random.seed(42)

def coordinate(dgm, dim = 100):
    dgm = dgm_filter(dgm)
    tmp = dgm2diag(dgm)
    coordinates = [p[0] for p in tmp] + [p[1] for p in tmp]
    tor = 1e-3
    try:
        assert -1 -tor <= min(coordinates)
        assert max(coordinates) <= 1 + tor
    except AssertionError:
        print('min and max of coordinates is %s and %s'%(min(coordinates), max(coordinates)))
        sys.exit()
    vec,_ = np.histogram(coordinates, bins=dim, range=[-1,1])
    return vec.reshape((1,dim))

def permute(dgm, seed=42, seed_flag = True):
    """
    :param dgm: diagram in dionysus form
    :param seed:
    :return:
    """

    # dgm = randomdgms(10)[0]
    if seed_flag:
        np.random.seed(seed)
    tmp = dgm2diag(dgm)  # tmp is a list of tuples
    coordinates = [p[0] for p in tmp] + [p[1] for p in tmp]
    np.random.shuffle(coordinates)
    assert len(coordinates) % 2 == 0

    n_removed = 0
    diag = []
    for i in range(0, len(coordinates), 2):
        b, d = coordinates[i], coordinates[i+1]
        if abs(b-d) < 1e-3: # dionysus does not allow points that is too close to diagonal
            n_removed += 1
            continue
        tmp = tuple((min(b, d), max(b, d)))
        diag.append(tmp)
    assert len(diag) + n_removed == len(coordinates)//2
    if len(diag) == 0:
        diag = [(0,0)]
    return diag2dgm(diag)

def array2dgm(x, fil_d = 'sub', print_flag = True):
    """ convert a array of shape (n,1) to a diagram where largest value is paired up with smallest value """

    assert x.shape[1] == 1
    x = x.tolist()
    x = [val for sublist in x for val in sublist]
    if len(x) % 2 == 1: x = x + [0]
    assert len(x) % 2 == 0

    if fil_d == 'sub':
        order = False
    elif fil_d == 'sup':
        order = True
    else:
        raise Exception(f'No fil_d {fil_d} in array2dgm')

    x.sort(reverse=order)
    lis = [] # a list of tuples
    while len(x)!=0:
        tuple = (x[0], x[-1])
        lis.append(tuple)
        x = x[1:-1]
    if print_flag:
        print('finish converting array to dgm...')
    return d.Diagram(lis)


# def permute_dgms(dgms):
#     res = []
#     for dgm in dgms:
#         res.append(permute(dgm))
#     return res

def fake_diagram(g, cardinality = 2, attribute='deg', seed=42, true_dgm = 'null'):
    random.seed(seed)
    sample_pool = nx.get_node_attributes(g, attribute).values()

    if true_dgm != 'null':
        tmp = dgm2diag(true_dgm) # tmp is array
        sample_pool = [p[0] for p in tmp] + [p[1] for p in tmp]

    try:
        sample = random.choice(sample_pool, size=2*cardinality, replace=False)
    except:
        sample = random.choice(sample_pool, size=2 * cardinality, replace=True)
    assert set(sample).issubset(set(sample_pool))

    dgm = []
    for i in range(0, len(sample),2):
        x_ = sample[i]
        y_ = sample[i+1]
        dgm.append((min(x_, y_), max(x_, y_)+1e-3))
    return d.Diagram(dgm)

def fake_diagrams(graphs_, dgms, true_dgms = ['null']*10000, attribute='deg', seed=45):
    fake_dgms = []
    for i in range(len(graphs_)):
        cardinality = len(dgms[i])
        if len(graphs_[i])==0:
             fake_dgms.append(d.Diagram([(0,0)]))
             continue
        tmp_dgm = fake_diagram(graphs_[i][0], cardinality = cardinality, attribute=attribute, seed=seed, true_dgm=true_dgms[i])
        fake_dgms.append(tmp_dgm)
    return fake_dgms

def permute_dgms(dgms, permute_flag = False, permute_ratio = 1, seed=42, seed_flag = True):
    """
    :param dgms: a list of dgm
    :param permute_flag: whether to permute or not
    :return:
    """
    permuted_dgms_list = []
    if permute_flag:
        if permute_ratio == 1:
            print('Permuting %s dgms' %str(len(dgms)))
            for dgm in dgms:
                dgm = permute(dgm, seed=seed, seed_flag=seed_flag)
                permuted_dgms_list.append(dgm)
            return permuted_dgms_list
        else:
            assert permute_ratio < 1
            n = len(dgms)
            permute_idx = random_.sample(range(n), int(n * permute_ratio))
            for i in range(n):
                if i in permute_idx:
                    dgm = permute(dgms[i], seed=seed, seed_flag=seed_flag)
                else:
                    dgm = dgms[i]
                permuted_dgms_list.append(dgm)
            return permuted_dgms_list

    else:
        return dgms



if __name__ == "__main__":
    dgm = d.Diagram([[1,2], [3,4], [5,6], [7,8]])
    from Esme.dgms.format import normalize_dgm
    dgm = normalize_dgm(dgm)
    x = coordinate(dgm, dim=20)
    print(x,x.shape)
    sys.exit()

    dgms = [dgm] * 10
    dgms = permute_dgms(dgms, permute_flag=True, permute_ratio=0.5)
    for dgm in dgms:
        print_dgm(dgm)
        print()