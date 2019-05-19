import numpy as np
from Esme.helper.format import precision_format as pf

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