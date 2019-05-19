import numpy as _np
import matplotlib.pyplot as _plt
from scipy.spatial import distance as _dist


def plot_graphon(graphon, resolution=1000):
    """Plot a graphon with matplotlib."""
    labels = _np.linspace(0, 1, resolution)
    heights = evaluate(graphon, labels)
    _plt.matshow(heights, cmap=_plt.cm.gray_r, vmin=0, vmax=1)
    ax = _plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def plot_adjacency(adj, **kwargs):
    _plt.matshow(adj, cmap=_plt.cm.gray_r, vmin=0, vmax=1, **kwargs)
    ax = _plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def sample_edges(probs):
    """Sample edges from an edge probability matrix."""
    n = len(probs)
    probs = _dist.squareform(probs, force='tovector', checks=False)
    coins = _np.random.sample(probs.shape)
    edges = coins <= probs
    edges = _dist.squareform(edges)
    edges[_np.diag_indices_from(edges)] = 0
    return edges.astype(int)


def evaluate(graphon, labels):
    """Evaluate a graphon at the labels."""
    ix_i, ix_j = _np.meshgrid(labels, labels)
    heights = graphon(ix_i.flatten(), ix_j.flatten())
    return heights.reshape((len(labels), -1))


def adaptive_neighborhood(n, c):
    return c * _np.sqrt(_np.log(n) / n)


def l_infinity_error(p, p_hat):
    p = _np.array(p)
    p_hat = _np.array(p_hat)

    _np.fill_diagonal(p, 0)
    _np.fill_diagonal(p_hat, 0)

    return _np.abs(p - p_hat).max()


def squared_l_2_error(p, p_hat):
    p = _np.array(p)
    p_hat = _np.array(p_hat)

    _np.fill_diagonal(p, 0)
    _np.fill_diagonal(p_hat, 0)

    n = len(p)

    return 1/n**2 * _np.linalg.norm(p - p_hat)**2
