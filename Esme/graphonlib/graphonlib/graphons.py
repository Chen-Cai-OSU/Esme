import numpy as _np
import numba as _numba


def make_constant_graphon(height):
    """Creates a graphon function with the specified height."""
    def constant_graphon(x, y):
        return height * _np.ones_like(x)
    
    return constant_graphon


def radial_graphon(x, y):
    """Construct a radial graphon function."""
    return 1 - _np.minimum(_np.sqrt(x**2 + y**2), 1)


@_numba.jit(nopython=True)
def _find_bucket(x, y, steps):
    i = 0
    while i < steps:
        midpoint = 1/(2**(i + 1))
    
        if x < midpoint and y < midpoint:
            i += 1
        elif x > midpoint and y > midpoint:
            i += 1
            x -= midpoint
            y -= midpoint
        else:
            break
    return i


@_numba.jit(nopython=True)
def _find_buckets_numba(x, y, steps, buckets):
    for i in range(len(buckets)):
        buckets[i] = _find_bucket(x[i], y[i], steps)
    return buckets


def _find_buckets(x, y, steps):
    x = _np.asarray(x)
    y = _np.asarray(y)
    steps = int(steps)
    
    buckets = _np.empty_like(x)
    _find_buckets_numba(x, y, steps, buckets)
    return buckets.squeeze().astype(int)


def make_hierarchical_blockmodel_graphon(n, heights):
    """Construct a hierarchical blockmodel graphon."""
    heights = _np.asarray(heights)
    def graphon(x, y):
        ix = _find_buckets(x, y, n)
        return heights[ix]
    return graphon


def make_discrete_graphon(arr):
    """Construct a graphon from a square, symmetric matrix."""
    n = len(arr)
    heights = _np.pad(arr, [(0,1), (0,1)], mode='edge')
    def graphon(x, y):
        ix = _np.floor(x * n).astype(int)
        iy = _np.floor(y * n).astype(int)
        return heights[ix, iy]
    return graphon


def make_tricky_graphon(low=.25, high=.75, trick_height=.75, trick_size=.05):
    def graphon(x, y):
        heights = _np.ones_like(x) * low

        regions = ((x <= .5) & (y <= .5)) | ((x > .5) & (y > .5))
        heights[regions] = high

        tricky_region = (
            (
                (.75 - trick_size/2 < x) & (x < .75 + trick_size/2) &
                (.25 - trick_size/2 < y) & (y < .25 + trick_size/2)
            ) |
            (
                (.25 - trick_size/2 < x) & (x < .25 + trick_size/2) &
                (.75 - trick_size/2 < y) & (y < .75 + trick_size/2)
            )
        )
        heights[tricky_region] = trick_height

        return heights
    return graphon
