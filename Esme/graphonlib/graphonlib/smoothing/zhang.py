from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import itertools as _itertools

import numba as _numba
import numpy as _np

from . import common as _common


def _distribute_assignments(n, k):
    """Assigns first n integers to k lists in a distributed fasion."""
    assignments = [list() for _ in range(k)]
    buckets = _itertools.cycle(assignments)

    for i in range(n):
        next(buckets).append(i)
        
    return assignments


@_numba.jit(nopython=True, nogil=True)
def _distance_matrix_core(args):
    """JIT-compiled core of adjacency matrix distance computation."""
    sqadj, d, assignment = args
    n = len(sqadj)
    for i in assignment:
        for j in range(i):
            max_k = None
            max_value = -_np.inf

            for k in range(n):
                if k == i or k == j:
                    continue
                else:
                    value = abs(sqadj[i, k] - sqadj[j, k])
                    if value > max_value:
                        max_value = value
                        max_k = k

            d[i,j] = d[j,i] = max_value


def distance_matrix(sqadj, n_threads=4):
    """Compared the distance matrix from the squared adjacency."""
    n = len(sqadj)
    d = _np.empty_like(sqadj)
    
    assignments = _distribute_assignments(n, n_threads)
    arglist = [(sqadj, d, a) for a in assignments]

    with _ThreadPoolExecutor(n_threads) as pool:
        res = list(pool.map(_distance_matrix_core, arglist))
    
    _np.fill_diagonal(d, _np.inf)
    return d


neighborhoods = _common.neighborhoods


def smoother(adj, h=.3, n_threads=4):
    """Helper function to compute the Zhang edge probability matrix."""
    adj = _np.asarray(adj, float)
    sqadj = adj.dot(adj)

    d = distance_matrix(sqadj, n_threads=n_threads)
    n = neighborhoods(d, h)
    return _common.smooth_neighborhoods(adj, n)
