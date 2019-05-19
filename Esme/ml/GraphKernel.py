
# coding: utf-8

# In[2]:
import sys
import torch
import collections
import sys
import time
import numpy as np
import scipy as sp
import scipy.linalg as linalg
from collections import OrderedDict
import Esme.ml.mnist as mnist
import keras
import numpy
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.model_selection import train_test_split


# ## Utilities

# In[3]:

def floatX(data):
    return np.float32(data)

get_mem_usage = lambda d, l, bs, n, s: ((d + 2 * l + bs) * n + s * 1000) * 4. / 1024**3

def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)


# ## Kernels

# In[4]:

def euclidean_distances(X, Y, squared=True):
    """ Calculate the pointwise distance.
    
    Arguments:
        X: of shape (n_sample, n_feature).
        Y: of shape (n_center, n_feature).
        squared: boolean.
    
    Returns:
        pointwise distances (n_sample, n_center).
    """
    X2 = torch.sum(X**2, dim=1, keepdim=True)
    if X is Y:
        Y2 = X2
    else:
        Y2 = torch.sum(Y**2, dim=1, keepdim=True)
    Y2 = torch.reshape(Y2, (1, -1))
    
    distances = X.mm(torch.t(Y))
    distances.mul_(-2)
    distances.add_(X2)
    distances.add_(Y2)
    if not squared:
        distances._sqrt()
    
    return distances

def Gaussian(X, Y, s):
    """ Gaussian kernel.
    
    Arguments:
        X: of shape (n_sample, n_feature).
        Y: of shape (n_center, n_feature).
        s: kernel bandwidth.
    
    Returns:
        kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0 
    K = euclidean_distances(X, Y)
    K.clamp_(min=0)
    gamma = 1. / (2 * s ** 2)
    K.mul_(-gamma)
    K.exp_()
    return K

def Laplacian(X, Y, s):
    """ Laplacian kernel.
    
    Arguments:
        X: of shape (n_sample, n_feature).
        Y: of shape (n_center, n_feature).
        s: kernel bandwidth.
    
    Returns:
        kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0
    K = euclidean_distances(X, Y, squared=False)
    K.clamp_(min=0)
    gamma = 1. / (2 * s ** 2)
    K.mul_(-gamma)
    K.exp_()
    return K

def Dispersal(X, Y, s, gamma):
    """ Dispersal kernel.
    
    Arguments:
        X: of shape (n_sample, n_feature).
        Y: of shape (n_center, n_feature).
        s: kernel bandwidth.
        gamma: dispersal factor.
    
    Returns:
        kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0

    K = euclidean_distances(X, Y)
    K.pow_(gamma / 2.)
    K.mul_(-1./ s)
    K.exp_()
    return K


# ## EigenPro

# In[5]:

def nystrom_kernel_svd(X, kernel_f, q):
    """Compute top eigensystem of kernel matrix using Nystrom method.

    Arguments:
        X: data matrix of shape (n_sample, n_feature).
        kernel_f: kernel tensor function k(X, Y).
        q: top-q eigensystem.
        bs: batch size.

    Returns:
        s: top eigenvalues of shape (q).
        U: (rescaled) top eigenvectors of shape (n_sample, q).
    """

    m, d = X.shape

    # Assemble kernel function evaluator.
    K = kernel_f(X, X).cpu().data.numpy()
    W = K / m
    w, V = sp.linalg.eigh(W, eigvals=(m-q, m-1))
    U1r, s = V[:, ::-1], w[::-1][:q]
    NU = floatX(U1r[:, :q] / np.sqrt(m))

    return s, NU


def pre_eigenpro_f(feat, phi, q, n, mG, alpha, min_q=5, seed=1):
    """Prepare gradient map f for EigenPro and calculate
    scale factor for step size such that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - f(g))

    Arguments:
        feat:   feature matrix.
        phi:    feature map or kernel function.
        q:      top-q eigensystem for constructing eigenpro iteration/kernel.
        n:      number of training points.
        mG:     maxinum batch size corresponding to GPU memory. 
        alpha:  exponential factor (<= 1) for eigenvalue ratio.
        min_q:  minimum value of q when q (if None) is calculated automatically.
        seed:   seed for random number.

    Returns:
        f:      tensor function.
        scale:  factor that rescales step size.
        s1:     largest eigenvalue.
        beta:   largest k(x, x) for the EigenPro kernel.
    """

    np.random.seed(seed) # set random seed for subsamples
    start = time.time()
    n_sample, d = feat.shape

    if q is None:
        svd_q = min(n_sample - 1, 1000)
    else:
        svd_q = q

    _s, _V = nystrom_kernel_svd(feat, phi, svd_q)
    
    # Choose k such that the batch size is bounded by
    #   the subsample size and the memory size.
    #   Keep the original k if it is pre-specified.
    qmG = np.sum(np.power(1 / _s, alpha) < min(n_sample / 5, mG)) - 1
    if q is None:
        max_m = min(max(n_sample / 5, mG), n_sample)
        q = np.sum(np.power(1 / _s, alpha) < max_m) - 1
        q = max(q, min_q)

    _s, _sq, _V = _s[:q-1], _s[q-1], _V[:, :q-1]

    s = torch.tensor(_s.copy()).to(feat.device)
    V = torch.tensor(_V).to(feat.device)
    sq = torch.tensor(_sq, dtype=torch.float).to(feat.device)

    scale = np.power(_s[0] / _sq, alpha)
    D = (1 - torch.pow(sq / s, alpha)) / s
    pre_f = lambda g, Km: torch.mm(
        V * D, torch.t(torch.mm(torch.mm(torch.t(g), Km), V)))
    s1 = _s[0]
    
    print("SVD time: %.2f, q: %d, adjusted q: %d, s1: %.2f, new s1: %.2e" %
          (time.time() - start, qmG, q, _s[0], s1 / scale))

    kxx = 1 - np.sum(_V ** 2, axis=1) * n_sample
    beta = np.max(kxx)

    return pre_f, floatX(scale), s1, floatX(beta)


# ## Model

# In[13]:

class FKR_EigenPro(nn.Module):
    def __init__(self, kernel, centers, dim, device="cuda"):
        super(FKR_EigenPro, self).__init__()
        self.pinned_list = []
        self.kernel = kernel
        self.n, self.D = centers.shape
        self.dim = dim
        self.device = device
        self.centers = self.tensor(centers, release=True)
        self.w = self.tensor(torch.zeros(self.n, dim), release=True)

    def __del__(self):
#         dump_tensors()
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")
        torch.cuda.empty_cache()

    def tensor(self, x, dtype=None, release=False):
        t = torch.tensor(x, dtype=dtype, requires_grad=False).to(self.device)
        if release:
            self.pinned_list.append(t)
        return t

    def kernel_matrix(self, x):
        K = self.kernel(x, self.centers)
        return K

    def forward(self, x, weight=None):
        if weight is None:
            weight = self.w
        K = self.kernel_matrix(x)
        p = K.mm(weight)
#         del K
        return p

    def primal_gradient(self, x, y, weight):
        p = self.forward(x, weight)
        g = p - y
        return g

    @staticmethod
    def _compute_bs_eta(bs, mG, beta, s1):
        if bs is None:
            bs = min(np.int32(beta / s1 + 1), mG)

        if bs < beta / s1 + 1:
            eta = bs / beta
        else:
            eta = 0.99 * 2 * bs / (beta + (bs - 1) * s1)
        return bs, floatX(eta)
    
    def eigenpro_iterate(self, samples, x_batch, y_batch, f,
                         eta, sample_ids, batch_ids):
        # update random coordiate block (for mini-batch)
        g = self.primal_gradient(x_batch, y_batch, self.w)
        self.w.index_add_(0, batch_ids, -eta * g)
        
        # update fixed coordinate block (for EigenPro)
        Km = self.kernel(x_batch, samples)
        e = f(g, Km)
        self.w.index_add_(0, sample_ids, eta * e)
        return
    
    def evaluate(self, x, y, bs):
        p_list = []
        n, _ = x.shape
        for batch_ids in np.array_split(range(n), n / min(n, bs)):
            x_batch = self.tensor(x[batch_ids])
            p_batch = self.forward(x_batch).cpu().data.numpy()
            p_list.append(p_batch)
        p = np.vstack(p_list)
        
        mse = np.mean(np.square(p - y))
        yc = np.argmax(y, axis=-1)
        pc = np.argmax(p, axis=-1)
        acc = np.mean(yc == pc)
        return mse, acc

    def fit(self, x_train, y_train, 
            x_val, y_val, epochs, mem_gb,
            n_subsample=None, q=None,
            bs=None, eta=None,
            scale=1, seed=1):     
        
        n, n_label = y_train.shape
        if n_subsample is None:
            if n < 100000:
                n_subsample = min(n, 2000)
            else:
                n_subsample = 12000
       
        mem_bytes = (mem_gb - 1) * 1024**3 # preserve 1GB
        bsizes = np.arange(n_subsample)
        mem_usages = ((self.D + 3 * n_label + bsizes + 1) * self.n + n_subsample * 1000) * 4
        mG = np.sum(mem_usages < mem_bytes) # device-dependent batch size
    
        
        # Calculate batch/step size for improved EigenPro iteration.
        np.random.seed(seed)
        sample_ids = np.random.choice(n, n_subsample, replace=False)
        sample_ids = self.tensor(sample_ids)
        samples = self.centers[sample_ids]
        eigenpro_f, gap, s1, beta = pre_eigenpro_f(
            samples, self.kernel, q, n, mG, alpha=.95, seed=seed)
        new_s1 = s1 / gap

        if eta is None:
            bs, eta = self._compute_bs_eta(bs, mG, beta, new_s1)
        else:
            bs, _ = self._compute_bs_eta(bs, mG, beta, new_s1)
        eta_ = scale * eta

        print("n_subsample=%d, mG=%d, eta=%.2f, bs=%d, s1=%.2e, beta=%.2f" %
              (n_subsample, mG, eta, bs, s1, beta))
        eta = self.tensor(scale * eta / bs, dtype=torch.float)
        
        # Subsample training data for fast estimation of training loss.
        index = np.random.choice(n, min(n, n_subsample), replace=False)
        x_sample, y_sample = x_train[index], y_train[index]

        res = dict()
        initial_epoch=0
        train_sec = 0 # training time in seconds
        
        for epoch in epochs:
            start = time.time()
            for _ in range(epoch - initial_epoch):
                epoch_ids = np.random.choice(n, int(n / bs * bs), replace=False)
                for batch_ids in np.array_split(epoch_ids, n / bs):
                    x_batch = self.tensor(x_train[batch_ids])
                    y_batch = self.tensor(y_train[batch_ids])
                    batch_ids = self.tensor(batch_ids)
                    self.eigenpro_iterate(samples, x_batch, y_batch, eigenpro_f,
                                          eta, sample_ids, batch_ids)
#                     x_batch, y_batch, batch_ids = None, None, None
                    del x_batch, y_batch, batch_ids
            train_sec += time.time() - start
            tr_score = self.evaluate(x_sample, y_sample, bs)
            tv_score = self.evaluate(x_val, y_val, bs)
            print("train error: %.2f%%\tval error: %.2f%% (%d epochs, %.2f seconds)\t"
                  "train l2: %.2e\tval l2: %.2e" %
                  ((1 - tr_score[1]) * 100, (1 - tv_score[1]) * 100, epoch, train_sec, tr_score[0], tv_score[0]))
            res[epoch] = (tr_score, tv_score, train_sec)
            initial_epoch = epoch

        return res

import numpy as np

def grid_search(eval_f, params, select_f=min):
    evals = []
    for param in params:
        evals.append((eval_f(param), param))
    
    best_val, best_param = evals[0]
    for val, param in evals[1:]:
        if best_val != select_f(val, best_val):
            best_val, best_param = val, param
    return best_val, best_param

def line_search(eval_f, left, right, select_f=min):
    param2eval = dict()
    
    def _search4(eval_f, left, right):
        assert left <= right
        cleft = np.int(left + (right - left)/3)
        cright = np.int(right - (right - left)/3)
        
        if right - left > 1 and right - left < 3:
            cleft = left + 1
        
        ps = [left, cleft, cright, right]
        print("search bandwidth list", ps)
        for p in ps:
            if p not in param2eval.keys():
                param2eval[p] = eval_f(p)
        min_eval = select_f([param2eval[p] for p in ps])
        
        if right - left < 3:
            if min_eval == param2eval[left]:
                return left
            elif min_eval == param2eval[cleft]:
                return cleft
            else:
                return right
            
        if min_eval == param2eval[left]:
            return _search4(eval_f, left, cleft)
        elif min_eval == param2eval[left]:
            return _search4(eval_f, cright, right)
        elif min_eval == param2eval[cleft]:
            return _search4(eval_f, left, cright)
        else: # == param2eval[cright]
            return _search4(eval_f, cleft, right)
        
    best_param = _search4(eval_f, left, right)
    best_val = param2eval[best_param]
    return best_val, best_param


def init_train_Gaussian(s, x_train, y_train, x_test, y_test, epochs, n_class, mem_gb, device='cuda'):
    print("s = " + str(s))
    kernel = lambda x, y: Gaussian(x, y, s=s)
    model = FKR_EigenPro(kernel, x_train, n_class, device=device)
    res = model.fit(x_train, y_train, x_test, y_test, epochs=epochs, mem_gb=mem_gb)
    print()
    del model
    return 1 - res[epochs[-1]][1][1]  # validation classification error


if __name__ == '__main__':
    def eigenpro(x=None, y=None, rs=10, bd=1):
        """
        :param x: np.array of shape (n_data, dim)
        :param y: np.array of shape (n_data,)
        :param rs: random seed
        :param bd: Gaussian kernel bandwidth
        :return:
        """

        x, y = np.random.random(((70000, 784))), np.random.randint(0,10, size=70000)
        n_class = len(np.unique(y))
        y = keras.utils.to_categorical(y, n_class)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=rs)
        x_train, y_train, x_test, y_test = x_train.astype('float32'), y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        kernel = lambda x,y: Gaussian(x, y, s=1)
        model = FKR_EigenPro(kernel, x_train, n_class, device=device)
        res = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=12)
        eval_f = lambda s: init_train_Gaussian(s, x_train[:10000], y_train[:10000], x_test, y_test, epochs=[10], n_class=n_class, mem_gb=12, device = device)

    eigenpro()
    sys.exit()

    try:
        best_val, best_param =  line_search(eval_f, 1, 5) # search kernel bandwidth in [1, 10]
    except numpy.linalg.linalg.LinAlgError:
        print('LinAlgError')
        best_param = 10
    print("best (Gaussian kenrel) bandwidth = " + str(best_param))

