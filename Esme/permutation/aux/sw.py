import sys
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/')
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/aux/')

import argparse
import numpy as np
import time
import os
import subprocess
from tools import diag2dgm, print_dgm, make_direct, diags2dgms, load_graph
from util import fake_diagram, fake_diagrams, partial_dgms, dgms_stats
from svm import evaluate_tda_kernel
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, MDS
import pickle
import matplotlib.pyplot as plt
import dionysus as d
from joblib import delayed, Parallel

def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        # print(pt),
        # print(type(pt))
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag

def precision_format(nbr, precision=1):
    # assert type(nbr)==float
    return  round(nbr * (10**precision))/(10**precision)

def assert_sw_dgm(dgms):
    # check sw_dgm is a list array
    # assert_sw_dgm(generate_swdgm(10))
    assert type(dgms)==list
    for dgm in dgms:
        assert np.shape(dgm)[1]==2

def sw(dgms1, dgms2, parallel_flag=False, kernel_type='sw', n_directions=10, bandwidth=1.0, K=1, p = 1):
    # print('kernel type is %s'%kernel_type)
    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))

    import sklearn_tda as tda
    if parallel_flag==False:
        if kernel_type=='sw':
            tda_kernel = tda.SlicedWassersteinKernel(num_directions=n_directions, bandwidth=bandwidth)
        elif kernel_type=='pss':
            tda_kernel = tda.PersistenceScaleSpaceKernel(bandwidth=bandwidth)
        elif kernel_type == 'wg':
            tda_kernel = tda.PersistenceWeightedGaussianKernel(bandwidth=bandwidth, weight=arctan(K, p))
        else:
            print ('Unknown kernel')

        diags = dgms1; diags2 = dgms2
        X = tda_kernel.fit(diags)
        Y = tda_kernel.transform(diags2)
        return Y

def sw_parallel(dgms1, dgms2,  kernel_type='sw', parallel_flag=True, n_directions=10, granularity=25, **kwargs):
    t1 = time.time()
    assert_sw_dgm(dgms1)
    assert_sw_dgm(dgms2)
    from joblib import Parallel, delayed
    n1 = len(dgms1); n2 = len(dgms2)
    kernel = np.zeros((n2, n1))

    if parallel_flag==False:         # used as verification
        for i in range(n2):
            kernel[i] = sw(dgms1, [dgms2[i]], kernel_type=kernel_type, n_directions=n_directions, bandwidth=kwargs['bw'])
    if parallel_flag==True:
        # parallel version
        kernel = Parallel(n_jobs=-1)(delayed(sw)(dgms1, dgms2[i:min(i+granularity, n2)], kernel_type=kernel_type,
                                                 n_directions=n_directions, bandwidth=kwargs['bw'], K=kwargs['K'],
                                                 p=kwargs['p']) for i in range(0, n2, granularity))
        kernel = (np.vstack(kernel))
    print('Finish computing %s kernel'%kernel_type)
    return (kernel/float(np.max(kernel)), precision_format(time.time()-t1, 1))

def dgms2swdgm(dgms):
    swdgms=[]
    for dgm in dgms:
        diag = dgm2diag(dgm)
        swdgms += [np.array(diag)]
    return swdgms

def gengerate_dgm(n):
    import numpy as np
    dgm = []
    for i in range(n):
        a = np.random.rand()
        b = np.random.rand()
        pt = (min(a, b), max(a, b))
        dgm += [pt]
    return dgm

def highpowconf(lis):
    # input: list of numbs
    # output: dgm format. pair smallest to largest.
    lis.sort()
    assert len(lis) % 2 == 0
    res = [] # a list of tuples
    while len(lis) > 0:
        res.append((lis[0], lis[-1]))
        lis = lis[1:-1]
    return d.Diagram(res)

def lowpowconf(lis, pertumation = 1e-3):
    # input: list of numbs
    # output: dgm format. (sort and pair neighboring vals)
    lis = [num + np.random.rand() * pertumation for num in lis]
    lis.sort()
    assert len(lis) % 2 == 0
    res = [] # a list of tuples
    while len(lis) > 0:
        res.append((lis[0], lis[1]))
        lis = lis[2:]
    return d.Diagram(res)

def dgmnum(dgm):
    # get a list of sorted coordinates from dgm
    res = []
    for p in dgm:
        res.append(p.birth)
        res.append(p.death)
    res.sort()
    return res

def lowdgms(dgms):
    res = []
    for dgm in dgms:
        res.append(lowpowconf(dgmnum(dgm)))
    return res

def highdgms(dgms):
    res = []
    for dgm in dgms:
        res.append(highpowconf(dgmnum(dgm)))
    return res
# highpowconf(dgmnum(gengerate_dgm(10)))
# print_dgm(highpowconf(range(10)))

def dgmfilter(dgm, threshold = 0.1, print_flag=False):
    res = []
    n = len(dgm)
    for p in dgm:
        if (p.death - p.birth) > threshold:
            res.append((p.birth, p.death))
    if print_flag:
        print('Before filtering there are %s points, after there are %s points'%(n, len(res)))
    return d.Diagram(res)

def dgmsfilter(dgms, threshold = 0.1, print_flag = False):
    res = []
    for dgm in dgms:
        dgm = dgmfilter(dgm, threshold = 0.1, print_flag = False)
        res.append(dgm)
    return res

def generate_dgms(n, n_pt = 50, sw_flag=True):
    # sw_flag, if turned on, the format is for sw kernel. Otherwise,
    # it is dionysus form
    dgms = [0] * n
    for i in range(n):
        dgms[i] = gengerate_dgm(n_pt)
    if sw_flag:
        return dgms
    else:
        return [d.Diagram(dgm) for dgm in dgms]

def random_direction(noise = 0.1):
    return np.array((np.random.rand(), np.random.rand())) * noise

def translate_dgm(dgm, direc = np.array([0.1, 0.1])):
    diag = dgm2diag(dgm)
    translate_array = np.array(diag) + direc
    return diag2dgm(translate_array.tolist())

def mono_trans(dgm, power = 1.01):
    diag = dgm2diag(dgm)
    mono_array = np.array(diag)**power
    return diag2dgm(mono_array.tolist())

def generate_data(dgm, seed=1, noise = 0.1, power=1.01, print_flag=False):
    translated_dgm = translate_dgm(dgm, direc = random_direction(noise=noise))
    mono_dgm = mono_trans(dgm, power=power)
    fake_dgm = fake_diagram(cardinality = len(translated_dgm), seed = seed, true_dgm = dgm)
    if print_flag:
        print_dgm(translated_dgm)
        print_dgm(fake_dgm)
    return translated_dgm, mono_dgm, fake_dgm

def dgmsio(obj, args, method='deg', save_flag=True):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/experiments/' \
             + args.graph + '/' + args.kerneltype + '/' + method + '/'
    filename = 'dgms_' + method + '.pickle'
    make_direct(direct)

    if save_flag:
        pickle_out = open(direct + filename, "wb")
        pickle.dump(obj, pickle_out)
        pickle_out.close()
    else:
        print('load existing dgms from %s...'%direct + filename)
        pickle_in = open(direct + filename, "rb")
        return pickle.load(pickle_in)

def delete_diagpts(dgm):
    diag = dgm2diag(dgm)
    diag = [p for p in diag if p[0]!=p[1]]
    return diag2dgm(diag)

def distance_matrix_i(i):
    assert 'dgms' in globals().keys()
    n = len(dgms)
    line_i = np.zeros((1, n))
    line_i_ = np.zeros((1, n))
    for j in range(i, n):
        line_i[0, j] = d.bottleneck_distance(dgms[i], dgms[j])
    for j in range(n - i, n):
        line_i_[0, j] = d.bottleneck_distance(dgms[n - 1 - i], dgms[j])
    return (line_i, line_i_)

def distance_matrix(dgms):
    # need to scale up
    assert len(dgms) % 2 == 0
    n = len(dgms)
    dist_matrix = np.zeros((n, n)) - 3.14
    X = Parallel(n_jobs=-1)(delayed(distance_matrix_i)(i) for i in range((len(dgms) / 2) + 1))
    for i in range(len(X)):
        dist_matrix[i] = X[i][0]
        dist_matrix[len(dgms) - i - 1] = X[i][1]
    return (dist_matrix + dist_matrix.T) / 2.0

def load_label(graph):
    if graph == 'imdb_binary':
        label = [1] * 500 + [2] * 500
    elif graph == 'reddit_binary':
        label = [1] * 500 + [2] * 1000 + [1] * 500
    elif graph == 'dd_test':
        label = [1] * 691 + [2] * 487
    elif graph == 'mutag':
        label = [1] * 125 + [2] * 63
    elif graph == 'ptc':
        label = [1] * 152 + [2] * 192
    elif graph == 'imdb_multi':
        label = [1] * 500 + [2] * 500 + [3] * 500
    elif graph == 'new_neuron':
        label = [1] * 147 + [2] * 17 + [3] * 123 + [4] * 94 + [2] * 78
    elif graph == 'old_neuron':
        label = [1] * 710 + [7] * 420
    else:
        raise Exception('No label for %s'%graph)
    return label

def mds_vis(dist_matrix, colors, n_components=2):
    assert np.shape(dist_matrix)[0] == len(colors)
    # dist_matrix = distance_matrix(dgms)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    pos = mds.fit(dist_matrix).embedding_
    plt.scatter(pos[:, 0], pos[:, 1], c=colors, s=10, lw=0,
                label='True Position')
    plt.show()
    plt.close()

def tsne_vis(dist_matrix, y, rs = 42, show_flag = False, save_flag=False, legend_flag = True,  **param):
    # dist_matrix = tfkernel
    # https://matplotlib.org/examples/color/named_colors.html
    colour_dict = {1: 'r', -1: 'b'}
    colour_dict = {1: 'r', -1: 'bisque', 2: 'b', -2: 'aqua', 3:'g', -3: 'greenyellow'}
    colours = [colour_dict[i] for i in y]
    assert np.shape(dist_matrix)[0] == len(colours)
    tsne = TSNE(metric='precomputed', verbose=0, random_state=rs)
    pos = tsne.fit_transform(dist_matrix)

    if legend_flag:
        # https://stackoverflow.com/questions/26558816/matplotlib-scatter-plot-with-legend/26559256
        import matplotlib.patches as mpatches
        classes = ['True1', 'Fake1', 'True2', 'Fake2', 'True3', 'False3']
        class_colours = ['r', 'bisque', 'b', 'aqua', 'g', 'greenyellow']
        recs = []
        for i in range(0, len(class_colours)):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
        plt.legend(recs, classes, loc=4)

    plt.scatter(pos[:, 0], pos[:, 1], c=colours, s=10, lw=0, label='True Position')
    # plt.title('TSNE viz for graph %s and kernel %s'%(param['graph'], param['kernel']))
    plt.axis('off')
    if save_flag: plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/viz/' + param['name'] + '.png')
    if show_flag: plt.show()
    # plt.show()
    plt.close()

# alldgms = dgms + fake_dgms
# dgms = [delete_diagpts(dgm) for dgm in alldgms]
# m = distance_matrix(dgms)
# colors = [1]*1000 + [2]*1000
def test_legend():
    import matplotlib.patches as mpatches
    classes = ['A','B','C']
    class_colours = ['r','b','g']
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
    plt.legend(recs,classes,loc=4)

    x = [1, 3, 4, 6, 7, 9]
    y = [0, 0, 5, 8, 8, 8]
    colours = ['r', 'r', 'b', 'g', 'g', 'g']
    plt.scatter(x, y, c=colours)
    plt.show()
    plt.close()

def phase_transition_data():
    dd = {}
    dd['ricci'] = [50, 58.4, 77.8, 85.4, 90.7, 94.3, 96.5, 97.8, 98.8, 99.2, 99.3] # seed 43 # 84.2 pts in dgm on average
    dd['deg'] =   [50, 54.6, 83.6, 87.8, 91.7, 93.6, 97.4, 98.7, 99.0, 99.3, 99.5] # seed 42 # 72.6
    dd['cc'] =    [50.0, 71.4, 85.3, 92.5, 95.1, 95.7, 98.2, 98.6, 99.1, 99.6, 99.6]         # 39.6 pts in dgm on average
    dd['ricci_zoom'] = [58.4, 58.3, 58.1, 59.3, 69.4, 71.6, 72.8, 75.2, 77.3, 78.0, 77.8] # from 0.1 to 0.2
    dd['deg_zoom'] =   [54.6, 54.1, 78.8, 80.1, 79.6, 79.3, 79.2, 80.2, 81.2, 83.5, 83.7] # from 0.1 to 0.2
    dd['cc_zoom'] =    [50, 93.2, 84.3, 57.5, 56.9, 56.7, 56.2, 60.1, 62.5, 65.9, 71.5] # from 0 to 0.1 # something wired

    reddit_binary = {}
    reddit_binary['ricci'] = [50, 51.8, 52.7, 74.9, 78.9, 82.0, 83.5, 85.1, 86.2, 86.4, 86.9] #seed 43
    reddit_binary['deg'] =   [50, 74.5, 81.0, 85.6, 88.6, 90.9, 92.1, 93.6, 94.7, 95.3, 95.7]
    reddit_binary['cc']  =   [50, 52.9, 55.0, 76.9, 81.9, 85.0, 87.4, 88.3, 89.8, 91.1, 91.8 ]
    reddit_binary['ricci_zoom'] = [52.7, 52.9, 53.2, 53.0, 53.0, 53.2, 53.4, 73.0, 73.8, 73.9, 74.9] # from 0.2 to 0.3
    reddit_binary['deg_zoom'] =   [50.0, 50.4, 50.9, 52.0, 51.9, 52.1, 52.0, 52.9, 51.9, 51.3, 74.5]  # from 0 to 0.1


    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 11).tolist()
    x1_ = x + np.linspace(0.1, 0.2, 11).tolist()
    plt.plot(x, dd['deg'], label='dd deg', linestyle = '-', color='b')
    plt.plot(x, dd['ricci'], label='dd ricci', linestyle='--', color = 'b')
    plt.plot(x, dd['cc'], label = 'dd cc', linestyle = '-.', color = 'b')

    plt.plot(x, reddit_binary['ricci'], label='reddit_binary ricci', linestyle = '-', color='r')
    plt.plot(x, reddit_binary['deg'], label='reddit_binary deg', linestyle = '--', color='r')
    plt.plot(x, reddit_binary['cc'], label='reddit_binary cc', linestyle = '-.', color='r')

    # plt.plot()
    plt.ylabel('Accuracy')
    plt.xlabel('s')
    plt.legend()
    plt.show()

def normalize_kernel(k):
    n = np.shape(k)[0]
    I = np.ones(k.shape) / np.float(n)
    return k - 2 * np.dot(I, k) + I.dot(k).dot(I)

def sample_dgm(dgm, sample_ratio=0.5):
    # dgm = dgms[1]
    n = len(dgm)
    n_new = int(n * sample_ratio) + 1
    diag = dgm2diag(dgm)
    sample_idx = np.random.choice(range(n), size = n_new, replace=False)
    assert len(set(sample_idx)) == n_new
    diag_sample = [diag[i] for i in sample_idx]
    return diag2dgm(diag_sample)

def sample_dgms(dgms, sample_ratio = 0.5):
    sub_dgms = []
    for dgm in dgms:
        sub_dgms.append(sample_dgm(dgm, sample_ratio=sample_ratio))
    return sub_dgms

def densityestimation(x, bw = 0.1, print_flag = True):
    """
    :param x: a list
    :param bw:
    :param print_flag:
    :return: array
    """
    from sklearn.neighbors import KernelDensity
    x = [0,0,0]
    assert type(x)==list
    x = np.array(x).reshape(len(x),1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x)
    x_plot = np.linspace(-1, 1, 200)
    x_plot = np.array(x_plot).reshape(len(x_plot), 1)
    log_dens = kde.score_samples(x_plot)
    if not print_flag:
        plt.plot(np.exp(log_dens))
        plt.show()
    return np.exp(log_dens)

def dgms_de(dgms):
    def dgm_de(dgm):
        x = dgmnum(dgm)
        return densityestimation(x, bw=0.1, print_flag=True)
    res = np.zeros((len(dgms), 200))
    for i in range(len(dgms)):
        res[i] = dgm_de(dgms[i])
    return res

def graphs_de(graphs, attr = 'deg', bw = 0.1):
    def graph_de(graph, attr = 'deg', bw = 0.1):
        import networkx as nx
        x = nx.get_node_attributes(graph, attr).values()
        return densityestimation(x, bw=bw, print_flag=True)
    res = np.zeros((len(graphs), 200))
    for i in range(len(graphs)):
        for j in range(len(graphs[i])):
            res[i] += graph_de(graphs[i][j], attr = attr, bw = bw)
    return res

def getdgms_frommem(args):
    dgms_to_save_ = dgmsio(None, args, method=args.method, save_flag=False)
    dgms_to_save = {}

    try:
        for key, val in dgms_to_save_.items():
            if key != 'graphs':
                dgms_to_save[key] = diags2dgms(val)
            else:
                dgms_to_save[key] = val
        (graphs, dgms, sub_dgms, super_dgms, epd_dgms) = dgms_to_save['graphs'], dgms_to_save['dgms'], dgms_to_save[
            'sub_dgms'], dgms_to_save['super_dgms'], dgms_to_save['epd_dgms']
    except AttributeError: # for trees
        assert len(dgms_to_save_)
        dgms = diags2dgms(dgms_to_save_)
    if len(dgms) == 1268:
        dgms = dgms[:711] + dgms[849:]
    return dgms

def dgmxy(dgm):
    dgmlist = dgm2diag(dgm) # dmglist is a list of tuple
    dgmlistx = [p[0] for p in dgmlist]
    dgmlisty = [p[1] for p in dgmlist]
    return (dgmlistx, dgmlisty)

class Dgm(d._dionysus.Diagram):
    def __init__(self, dgm):
        self.dgm = dgm
        self.length = len(self.dgm)
        self.dgmnum()

    def stat(self, power=1):
        self.length = len(self.dgm)
        self.highpowconf()
        self.lowpowconf()
        self.randomconf()
        name = ['dgm', 'lowdgm', 'highdgm', 'randgm']
        assert len(self.dgm)==len(self.lowdgm)==len(self.highdgm)==len(self.randgm)
        i, stat = 0, []
        for dgm in [self.dgm, self.lowdgm, self.highdgm, self.randgm]:
            # print(name[i], self.energy(dgm, power=1), self.energy(dgm, power=2))
            stat.append(self.energy(dgm, power=power))
            i +=1
        return np.array(stat).reshape(1,4)

    def dgmnum(self):
        # get a list of sorted coordinates from dgm
        res = []
        for p in self.dgm:
            res.append(p.birth)
            res.append(p.death)
        res.sort()
        self.num =  res

    def dgmfilter(self, threshold=0.1, print_flag=False):
        res = []
        for p in self.dgm:
            if (p.death - p.birth) > threshold:
                res.append((p.birth, p.death))
        if print_flag:
            print('Before filtering there are %s points, after there are %s points' % (self.length, len(res)))
        self.dgm =  d.Diagram(res)

    def highpowconf(self):
        # input: list of numbs
        # output: dgm format. pair smallest to largest.
        lis = self.num
        lis.sort()
        assert len(lis) % 2 == 0
        res = []  # a list of tuples
        while len(lis) > 0:
            res.append((lis[0], lis[-1]))
            lis = lis[1:-1]
        self.highdgm =  d.Diagram(res)

    def lowpowconf(self, pertumation=1e-3):
        # input: list of numbs
        # output: dgm format. (sort and pair neighboring vals)
        lis = self.num
        lis = [num + np.random.rand() * pertumation for num in lis]
        lis.sort()
        assert len(lis) % 2 == 0
        res = []  # a list of tuples
        while len(lis) > 0:
            res.append((lis[0], lis[1]))
            lis = lis[2:]
        self.lowdgm =  d.Diagram(res)

    def randomconf(self):
        lis = self.num
        np.random.shuffle(lis)
        assert len(lis) % 2 == 0
        res = []  # a list of tuples
        while len(lis) > 0:
            res.append((lis[0], lis[1]))
            lis = lis[2:]
        self.randgm = d.Diagram(res)

    @staticmethod
    def energy(dgm, power = 1):
        energy = 0
        for p in dgm:
            energy += np.power(abs(p.birth - p.death), power)
        return energy

def movie(args, label):
    for dgmid in range(len(label)):
        direct = os.path.join('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/viz/', args.graph, args.method, args.kerneltype, 'movie')
        image_direct = os.path.join(direct, 'image')
        make_direct(direct)
        make_direct(image_direct)

        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.scatter(dgmxy(dgms[dgmid])[0], dgmxy(dgms[dgmid])[1], alpha = 0.2)
        plt.xlim(-0.1, 1.1); plt.ylim(-0.1, 1.1)
        plt.title('%s-th True diagram'%dgmid)

        plt.subplot(2, 2, 2)
        plt.scatter(dgmxy(fake_dgms[dgmid])[0], dgmxy(fake_dgms[dgmid])[1], alpha = 0.2)
        plt.xlim(-0.1, 1.1); plt.ylim(-0.1, 1.1)
        plt.title('%s-th Fake diagram' % dgmid)

        plt.savefig(image_direct + '/' + str(dgmid).zfill(4) + '.png')
    bashCommand = "ffmpeg -f image2 -r 3 -i ./image/%04d.png -vcodec mpeg4 -y movie.mp4"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd=direct)

parser = argparse.ArgumentParser()
parser.add_argument('--graph', default='imdb_binary', help="mutag, ptc, reddit...")
parser.add_argument('--kerneltype', default='sw', help="sw or pss")
parser.add_argument('--method', default='deg', help="deg, ricciCurvature or cc")

if __name__ == '__main__':
    # sys.argv = []
    args = parser.parse_args()
    graph = args.graph
    print(args.graph)
    x, y, y_coarse, y_finer = [], [], [], []
    label = load_label(graph)
    dgms = getdgms_frommem(args)

    stats1, stats2 = np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])
    for i in range(len(dgms)):
        x = Dgm(dgms[i])
        stats1 = np.concatenate((stats1, x.stat(power=1)), axis=0)
        stats2 = np.concatenate((stats2, x.stat(power=2)), axis=0)

    plt.figure()
    plt.subplot(211)
    plt.title('%s: Life time engergy. Ave pd pts: %.1f and %.1f'%(graph, dgms_stats(dgms)['ave'], 0.5 * dgms_stats(dgms)['ave_distinct']))
    plt.boxplot(stats1[1:], labels=['dgm', 'low', 'high', 'random'])
    plt.subplot(212)
    plt.boxplot(stats2[1:], labels=['dgm', 'low', 'high', 'random'])
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/viz/boxplot/'
    plt.savefig(direct + graph + '_' + args.method)
    # plt.show()
    sys.exit()



    hdgms = highdgms(dgms)
    ldgms = lowdgms(dgms)
    sample_dgms = sample_dgms(dgms, sample_ratio=0.5)
    dgms_control = generate_dgms(len(dgms), n_pt=int(dgms_stats(dgms)['ave'])+1 , sw_flag=False)
    fake_dgms = fake_diagrams(dgms, true_dgms=dgms, seed=45) # seed45

    movie(args, label)

    precisions = []
    for x in [hdgms, ldgms, fake_dgms, dgms]:
        tmp = []
        for bw_ in [1]:
            tfkernel, _ = sw_parallel(dgms2swdgm(x), dgms2swdgm(x), kernel_type='sw', bw=bw_, K=1, p=1)
            (precision, std, kparam, t) = evaluate_tda_kernel(tfkernel, np.array(label), (0, 0, {}, 0))
            tmp.append(precision)
        precisions.append(tmp)
    print precisions

    # fig, ax = plt.subplots(nrows=2, ncols=2)


    plt.subplot(2, 2, 3)
    d.plot.plot_diagram(dgms[1], show=True)
    plt.subplot(2, 2, 4)
    d.plot.plot_diagram(fake_dgms[1], show=True)
    plt.show()

    kernels = []
    for dgms_ in [dgms, dgms_control, fake_dgms, sample_dgms]:
        k, _ = sw_parallel(dgms2swdgm(dgms_), dgms2swdgm(dgms_), kernel_type=args.kerneltype, bw=1, K=1, p=1)
        k = normalize_kernel(k)
        kernels.append(k)

    spectrums = []
    for k in kernels:
        transformer = KernelPCA(n_components=20, kernel='precomputed')
        X_transformed = transformer.fit_transform(k)
        spectrums.append(transformer.lambdas_)
    print spectrums
    plt.plot(list(spectrums[0]), 'b', label='true diagrams')
    plt.plot(list(spectrums[1]), 'r', label='random diagram(uniform sampling)')
    plt.plot(list(spectrums[2]), 'y', label='fake diagrams')
    plt.plot(list(spectrums[3]), 'g', label='sample true diagrams')
    plt.yscale('linear')
    plt.legend()
    plt.title('Spectrum of ' + args.kerneltype + ' kernel for ' + args.graph + ' ' + args.method)
    print ('Finish plotting')
    plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/viz/spectrum/'
                + args.method + '_' + args.graph + '_' + args.kerneltype + '.png')
    sys.exit()
    for s in [1] :
    # for s in [0.001, 0.01] + np.linspace(0, 1, 11).tolist() + np.linspace(0.1, 0.2, 11).tolist():
        semifake_dgms = partial_dgms(dgms, portion = s, seed=42)
        x = dgms + semifake_dgms
        y = [1] * len(dgms) + [2] * len(dgms)
        tfkernel_sw, _ = sw_parallel(dgms2swdgm(x), dgms2swdgm(x), kernel_type='sw', bw=1, K=1, p=1)
        precision, std, kparam, time_ = evaluate_tda_kernel(tfkernel_sw, np.array(y), (0, 0, {}, 0), print_flag='off')
        print(s, precision)
        continue
        plt.matshow(np.sqrt(2 - 2 * tfkernel_sw))
        plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/viz/matrix/' + graph + '_' +  str(s) + '_distance.png')
        n = len(dgms)
        assert np.shape(tfkernel_sw) == (2*n, 2*n)
        plt.matshow(tfkernel_sw[0:n, 0:n] - tfkernel_sw[n:2*n, n:2*n])
        plt.savefig('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/viz/matrix/' + graph + '_' +  str(s) + '_diff.png')
        continue

        tsne_vis(2 - 2 * tfkernel_sw, y, rs=42, save_flag=True, legend_flag=False, **{'name': graph + '_' + str(s) + '_' + str(precision)})
        print ('When portion is %s, precison is %s'%(s, precision))

    sys.exit()
    for i in range(len(dgms)):
        dgm = dgms[i]
        fdgm = fake_dgms[i]
        x.append(dgm)
        x.append(fdgm)
        y += [1, -1]
        if label[i] == 1:
            y_finer += [1, -1]
            y_coarse += [1, 1]
        elif label[i] == 2:
            y_finer += [2, -2]
            y_coarse += [2, 2]
        elif label[i] == 3:
            y_finer += [3, -3]
            y_coarse += [3, 3]
        else:
            raise Exception('No such label')



    for bw in [6,7,8,9,10]:
        for dgms_ in [fake_dgms]:
            dgms_ = dgms_[1 - 1:711]+ dgms_[849:1269]
            trueswkernel, _ = sw_parallel(dgms2swdgm(dgms_), dgms2swdgm(dgms_), kernel_type='sw', bw=bw, K=1, p=1)
            precision, std, kparam, _ = evaluate_tda_kernel(trueswkernel, np.array(label), (0, 0, {}, 0), print_flag='on')
            print(precision, kparam)




    tfkernel_sw, _ = sw_parallel(dgms2swdgm(x), dgms2swdgm(x), kernel_type='sw', bw=1, K=1, p=1)
    tfkernel_pss, _ = sw_parallel(dgms2swdgm(x), dgms2swdgm(x), kernel_type='pss', bw=1, K=1, p=1)
    x_no_diagonal = [delete_diagpts(dgm) for dgm in x]
    dgms = x_no_diagonal
    bd_distance = distance_matrix(dgms)

    if True:
        precision, std, kparam, time = evaluate_tda_kernel(tfkernel_sw, np.array(y), (0, 0, {}, 0), print_flag='on')
        print('True from Fake: Using SW kernel can get accuracy %s' % precision)
        precision, std, kparam, time = evaluate_tda_kernel(tfkernel_sw, np.array(y_finer), (0, 0, {}, 0), print_flag='on')
        print('Finer: Using SW kernel can get accuracy %s' % precision)
        precision, std, kparam, time = evaluate_tda_kernel(tfkernel_sw, np.array(y_coarse), (0, 0, {}, 0), print_flag='on')
        print('Coarser: Using SW kernel can get accuracy %s' % precision)


        plt.subplot(2, 1, 1)
        plt.title('TSNE viz for graph %s and kernel sw and pss' % (graph))
        tsne_vis(2 - 2 * tfkernel_sw, y_finer, rs=42)
        plt.subplot(2, 1, 2)
        # tsne_vis(2 - 2 * tfkernel_pss, y_finer, rs=42)
        tsne_vis(bd_distance, y_finer, rs=42)
        plt.show()
        # mds_vis(2 - 2 * tfkernel_sw, y)

        sys.exit()

        tsne_vis(2 - 2 * tfkernel_pss, y)
        mds_vis(2 - 2 * tfkernel_pss, y)

    # do not differentiate true from fake diagrams
    precision, std, kparam, time = evaluate_tda_kernel(tfkernel_sw, np.array(y_coarse), (0, 0, {}, 0), print_flag='on')
    print(precision, kparam, time)

    # differentiate true from fake diagrams
    precision, std, kparam, time = evaluate_tda_kernel(tfkernel_sw, np.array(y_finer), (0, 0, {}, 0), print_flag='on')
    print(precision, kparam, time)
    sys.exit()


    (precision, std, kparam, time) = evaluate_tda_kernel(tfkernel_sw, np.array(y_finer), (0, 0, {}, 0), print_flag='on')
    print precision
    print kparam
    print time
    sys.exit()
    # dgm = d.Diagram([(.1,.3),(.2,.7), (.5, .6), (.1, .7), (.3, .6), (.2, .8), (.7, .9), (.1, .8)] + [(.1, .3)]*50 + [(.2, .8)]*50)
    for i in range(500):
        dgm = d.Diagram(gengerate_dgm(50))
        tdgm, mdgm, fdgm = generate_data(dgm, noise=.3, power=1, seed=i)
        tdgm_, mdgm_, fdgm_ = generate_data(dgm, noise=.3, power=1, seed=i**2)
        x.append(fdgm)
        x.append(dgm)
        # x.append(fdgm_)
        # y.append(1)
        y.append(1)
        y.append(-1)
    tfkernel, _ = sw_parallel(dgms2swdgm(x), dgms2swdgm(x), kernel_type='sw', bw=1, K=1, p=1)
    (precision, std, kparam, time) = evaluate_tda_kernel(tfkernel, np.array(y), (0, 0, {}, 0))
    print precision
    sys.exit()
    # dgms = generate_dgms(2000)
    # %time k = sw_parallel(dgms, dgms, kernel_type='pss', p=1, bw=1, K=1)