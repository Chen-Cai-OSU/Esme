import numpy as _np

from . import common as _common
from . import zhang as _zhang


def delete_independent_node(sqadj, adj, k):
    """Delete the independent node k from the squared adjacency.

    If sqadj is adj @ adj, and adj' is adj with the k row and column set to 0,
    this computes adj' * adj' efficiently  (i.e., without repeating the matrix 
    multiplication).
    
    """
    corrected = sqadj - _np.outer(adj[k], adj[k])
    corrected[k] = corrected[:,k] = 0
    return corrected


def distance_matrix(sqadj, k, n_threads=4):
    """Compute the distance matrix without observing node k."""
    d_k = _zhang.distance_matrix(sqadj, n_threads=n_threads)
    d_k[k] = d_k[:,k] = _np.inf
    return d_k


def neighborhoods(d_k, k, h):
    """Compute the neighborhoods without observing node k."""
    n_k = _common.neighborhoods(d_k, h)
    n_k[k] = n_k[:,k] = False
    return n_k


def _smooth(adj, n_k, k):
    p_hat = _np.zeros(len(adj))
    for i in range(len(adj)):
        if i == k:
            p_hat[i] = 0
            continue
        p_hat[i] = adj[n_k[i], k].mean()
    return p_hat


def smoother(adj, h=0.3, n_threads=4):
    adj = _np.asarray(adj, float)
    sqadj = adj.dot(adj)
    p_hat = _np.zeros_like(sqadj)

    for k in range(len(adj)):
        corrected = delete_independent_node(sqadj, adj, k)
        d_k = distance_matrix(sqadj, k, n_threads=n_threads)
        n_k = neighborhoods(d_k, k, h)
        smoothed = _smooth(adj, n_k, k)
        p_hat[:,k] += smoothed

    return (p_hat + p_hat.T)/2
