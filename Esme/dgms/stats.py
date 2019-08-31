""" functons related to statistics of persistence diagrams. """
import dionysus as d
import numpy as np
from Esme.helper.format import precision_format as pf
import networkx as nx

def normalize_(X, axis=0):
    from sklearn.preprocessing import normalize
    return normalize(X, axis=axis)

def dgms_summary(dgms, debug='off'):
    n = len(dgms)
    total_pts = [-1]*n
    unique_total_pts = [-1] * n # no duplicates
    for i in range(len(dgms)):
        total_pts[i] = len(dgms[i])
        unique_total_pts[i] = len(set(list(dgms[i])))
    if debug == 'on':
        print('Total number of points for all dgms')
        print(dgms)
    stat_with_multiplicity = (pf(np.mean(total_pts), precision=1), pf(np.std(total_pts), precision=1), np.min(total_pts), np.max(total_pts))
    stat_without_multiplicity = (pf(np.mean(unique_total_pts)), pf(np.std(unique_total_pts)), np.min(unique_total_pts), np.max(unique_total_pts))
    print('Dgms with multiplicity    Mean: %s, Std: %s, Min: %s, Max: %s'%(pf(np.mean(total_pts)), pf(np.std(total_pts)), pf(np.min(total_pts)), pf(np.max(total_pts))))
    print('Dgms without multiplicity Mean: %s, Std: %s, Min: %s, Max: %s'%(pf(np.mean(unique_total_pts)), pf(np.std(unique_total_pts)), pf(np.min(unique_total_pts)), pf(np.max(unique_total_pts))))
    return (stat_with_multiplicity, stat_without_multiplicity)

def test_():
    import dionysus as d
    dgm = d.Diagram([(1,1),(1,2)])
    dgm_ = d.Diagram([(1,1)])
    dgms = [dgm] * 10 + [dgm_] * 3
    dgms_summary(dgms)

def viz_dgm():
    import sklearn_tda
    sklearn_tda.persistence_image()

def print_dgm(dgm):
    for p in dgm:
        print(p)

def dgm_filter(dgm):
    """ if input is an empyt dgm, add origin point """
    if len(dgm) > 0:
        return dgm
    else:
        return d.Diagram([[0,0]])

def stat(lis, high_order=False):
    # lis = [a for a in lis if a!=0.00123]
    # list = angledgm
    if high_order == True:
        pass
    return np.array([np.min(lis), np.max(lis), np.median(lis), np.mean(lis), np.std(lis)])

def bl0(dgms_to_save_, key='deg'):
    # todo refactor
    graphs = dgms_to_save_['graphs']
    n = len(graphs)
    blfeat = np.zeros((n, 5))
    for i in range(n):
        fval = nx.get_node_attributes(graphs[i][0], key).values()
        blfeat[i] = stat(fval)
    return blfeat

def bl1(dgms):
    """ for each dgm in dgms, compute a 5 dim feature vector (min, max, mean, median, std) """
    n = len(dgms)
    blfeat = np.zeros((n, 5))

    for i in range(n):
        dgm = dgms[i]  # a list of lists
        cval = []
        for p in dgm:
            cval += p
        assert len(cval) == 2 * len(dgm)
        blfeat[i] = stat(cval)

    blfeat = normalize_(blfeat, axis=0)
    return blfeat

