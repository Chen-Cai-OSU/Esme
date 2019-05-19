import numpy as _np

from . import common as _common
from . import zhang as _zhang


def hpows(m, p):
    """Fractional power of hermitian matrix."""
    eigvals, eigvecs = _np.linalg.eigh(m)
    eigvals = _np.diag(eigvals).astype('complex')**p
    return eigvecs.dot(eigvals).dot(eigvecs.T)


def distance_matrix(m):
    q = m.conj().dot(m)
    s = (q.diagonal() + q.diagonal()[:,None]) - q - q.T
    return _np.sqrt(s.real)


neighborhoods = _common.neighborhoods


def smoother(adj, h=.3, power=2):
    d = hpows(adj.astype(float), power)
    s = distance_matrix(d)
    n = neighborhoods(s, h)
    return _common.smooth_neighborhoods(adj, n)
