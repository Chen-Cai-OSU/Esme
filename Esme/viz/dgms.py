import networkx as nx
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

from Esme.dgms.format import dgms2swdgms
from Esme.dgms.test import randomdgms
import dionysus as d
from Esme.dgms.vector import dgms2feature, merge_dgms


def density(data):
    # https: // www.data - to - viz.com / graph / density2d.html
    # Create data: 200 points
    # data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)

    x, y = data.T
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))

    axes[0].set_title('Scatterplot')
    axes[0].plot(x, y, 'ko')
    nbins = 20
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # plot a density
    axes[1].set_title('Calculate Gaussian KDE')
    axes[1].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)

    # # add shading
    axes[2].set_title('2D Density with shading')
    axes[2].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    plt.show()

def viz_vector():
    # https: // matplotlib.org / users / pyplot_tutorial.html
    dgm = d.Diagram([(2, 3), (3, 4)])
    from Esme.dgms.format import dgmxy
    dgmx, dgmy = dgmxy(dgm)
    dgms = [dgm] * 2

    params = {'bandwidth': 1.0, 'weight': (1, 1), 'im_range': [0, 1, 0, 1], 'resolution': [5, 5]}
    image = dgms2feature(dgms, vectype='pi', **params)
    images = merge_dgms(dgms, dgms, vectype='pi', **params)
    print(np.shape(image), np.shape(images))

    plt.figure()
    plt.subplot(121)
    plt.scatter(dgmx, dgmy)
    plt.subplot(122)
    plt.plot(images.T) # (n_image, dim)
    plt.show()


if __name__ == '__main__':
    viz_vector()
    sys.exit()
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    density(data)
    dgms = randomdgms(10)
    swdgms = dgms2swdgms(dgms)
    onedgm = np.vstack(swdgms)
    density(onedgm)