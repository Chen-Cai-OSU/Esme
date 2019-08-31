import networkx as nx
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

from Esme.dgms.format import dgms2swdgms
from Esme.dgms.test import randomdgms
import dionysus as d
from Esme.dgms.vector import dgms2vec, merge_dgms
from Esme.graph.dataset.tu_dataset import load_tugraphs
from Esme.dgms.fil import gs2dgms, gs2dgms_parallel
from Esme.dgms.fake import permute_dgms
from Esme.dgms.kernel import sw, sw_parallel
from Esme.dgms.format import print_dgm


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
    image = dgms2vec(dgms, vectype='pi', **params)
    images = merge_dgms(dgms, dgms, vectype='pi', **params)
    print(np.shape(image), np.shape(images))

    plt.figure()
    plt.subplot(121)
    plt.scatter(dgmx, dgmy)
    plt.subplot(122)
    plt.plot(images.T) # (n_image, dim)
    plt.show()


if __name__ == '__main__':

    # fake fake test
    graph = 'imdb_binary' # 'reddit_binary'
    norm = True
    fil = 'ricci'
    gs, labels = load_tugraphs(graph)
    # subdgms = gs2dgms(gs, fil=fil, fil_d='sub', norm=norm, graph = graph, ntda = False, debug_flag = False)
    subdgms = gs2dgms_parallel(gs, fil=fil, fil_d='sub', norm=norm, graph=graph, ntda=False, debug_flag=False)

    true_dgms = subdgms
    fake_dgms = permute_dgms(true_dgms, permute_flag=True, seed=42)
    another_fake_dgms = permute_dgms(true_dgms, permute_flag=True, seed=41)

    print_dgm(true_dgms[0])
    print('-'*20)
    print_dgm(fake_dgms[0])
    print('-' * 20)
    print_dgm(another_fake_dgms[0])

    all_dgms = true_dgms + fake_dgms
    all_dgms = dgms2swdgms(all_dgms)

    feat_kwargs = {'n_directions': 10, 'bw':1}
    k, _ = sw_parallel(all_dgms, all_dgms, parallel_flag=True, kernel_type='sw', **feat_kwargs)
    from Esme.viz.matrix import viz_distm
    fake_labels = [-label for label in labels]
    viz_distm(k, mode='tsne', y= labels + fake_labels)

    print(k.shape)

    sys.exit()
    viz_vector()
    sys.exit()
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    density(data)
    dgms = randomdgms(10)
    swdgms = dgms2swdgms(dgms)
    onedgm = np.vstack(swdgms)
    density(onedgm)