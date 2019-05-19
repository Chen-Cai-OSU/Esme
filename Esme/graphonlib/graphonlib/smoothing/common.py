import numpy as _np


def neighborhoods(d, h):
    """Compute the neighborhood membership matrix."""
    cutoffs = _np.percentile(d, 100*h, axis=0)
    n = d <= cutoffs[:, _np.newaxis]
    _np.fill_diagonal(n, 0)
    return n


def smooth_neighborhoods(adj, n):
    """Smooth the adjacency matrix with respect to the neighborhoods."""
    p_hat = []
    for i, neighbors in enumerate(n):
        x = adj[neighbors].mean(axis=0)
        p_hat.append(x)

    p_hat = _np.vstack(p_hat)
    p_hat = 1/2 * (p_hat + p_hat.T)
    _np.fill_diagonal(p_hat, 0)
    return p_hat
