import numpy as np
import networkx as nx
import dionysus as d
import time
import sys
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
import sklearn_tda as tda

from .sw import dgms2swdgm
from .tools import dgms_summary, add_dgms, precision_format, evaluate_best_estimator, remove_zero_col, normalize_, dgms2diags
from .svm import clf_search_offprint, rfclf
from .util import bl1, stat

def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag
def dgms2diags(dgms):
    diags = []
    for i in range(len(dgms)):
        diags.append(dgm2diag(dgms[i]))
    return diags

def filterdict(dict, keys):
    # keys: a list of keys
    dict_ = {}
    for key, val in dict.items():
        if key in keys: dict_[key] = val
    return dict_

def unwrap_pdvector(*arg, **kwarg):
    # http://qingkaikong.blogspot.com/2016/12/python-parallel-method-in-class.html
    return pdvector.persistence_vector(*arg, **kwarg)

class pipl():
    def __init__(self, dgms, type='pi'):
        self.dgms = dgms
        self.vec_type = type
        self.diags = self.dgms2swdgm(dgms)
        self.axis = 1

    def dgms2swdgm(self, dgms):
        swdgms = []
        for dgm in dgms:
            diag = self.dgm2diag(dgm)
            swdgms += [np.array(diag)]
        return swdgms

    def dgm2diag(self, dgm):
        assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
        diag = list()
        for pt in dgm:
            if str(pt.death) == 'inf':
                diag.append([pt.birth, float('Inf')])
            else:
                diag.append([pt.birth, pt.death])
        return diag

    def dgms_vecs(self, **kwargs):
        self.param = kwargs
        print kwargs
        t1 = time.time()
        def arctan(C, p):
            return lambda x: C * np.arctan(np.power(x[1], p))

        if self.vec_type == 'pi':
            diagsT = tda.DiagramPreprocessor(use=True, scaler=tda.BirthPersistenceTransform()).fit_transform(self.diags)

            kwargs = filterdict(kwargs, ['bandwidth', 'weight', 'im_range', 'resolution'])
            kwargs['weight'] = arctan(kwargs['weight'][0], kwargs['weight'][1])

            PI = tda.PersistenceImage(**kwargs)
            # PI = tda.PersistenceImage(bandwidth=1.0, weight=arctan(1.0, 1.0), im_range=[0, 1, 0, 1], resolution=[25, 25])
            res = PI.fit_transform(diagsT)

        elif self.vec_type == 'pl':
            kwargs = filterdict(kwargs, ['num_landscapes', 'resolution'])
            LS = tda.Landscape(**kwargs)
            # LS = tda.Landscape(num_landscapes=5, resolution=100)
            res = LS.fit_transform(self.diags)
        else:
            raise Exception('Unknown vec_type. You can only chose pi or pl')

        t2 = time.time()
        t = precision_format((t2 - t1), 1)
        self.t = t
        return remove_zero_col(normalize_(res, axis=self.axis), cor_flag=False)

    def summary(self):
        return {'time': self.t, 'param': self.param, 'vec_type': self.vec_type}

class pdvector():
    def __init__(self, dynamic_range_flag = True):
        self.dynamic_range_flag = dynamic_range_flag

    def data_interface(self, dgm, dynamic_range_flag=True):
        # from dgm to data/max/min
        for p in dgm: assert p.death >= p.birth
        data = [tuple(i) for i in dgm2diag(dgm)]
        try:
            [list1, list2] = zip(*data);
        except:
            print('Problem')
            list1 = [0];
            list2 = [1e-5]  # adds a dummy 0

        if dynamic_range_flag == True:
            min_ = min(min(list1), min(list2))
            max_ = max(max(list1), max(list2))
            std_ = (np.std(list1) + np.std(list2)) / 2.0
        elif dynamic_range_flag == False:
            min_ = -5
            max_ = 5
            std_ = 3

        return {'data': data, 'max': max_ + std_, 'min': min_ - std_}

    @staticmethod
    def rotate_data(data, super_check):
        """
        :param data:
        :return: a list of tuples
        """

        def rotate(x, y):
            return np.sqrt(2) / 2 * np.array([x + y, -x + y])

        def flip(x, y):
            assert x >= y
            return np.array([y, x])

        length = len(data)
        rotated = []

        for i in range(0, length, 1):
            if super_check == True: data[i] = flip(data[i][0], data[i][1])
            point = rotate(data[i][0], data[i][1]);
            point = (point[0], point[1])
            rotated.append(point)
        return rotated
    @staticmethod
    def draw_data(data, imax, imin, discrete_num=500):
        """
        :param data: a list of tuples
        :return: a dictionary: vector of length 1000

        """
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2) + 0.000001))

        discrete_num = discrete_num
        assert imax >= imin
        distr = np.array([0] * discrete_num)
        par = data
        for x, y in par:
            mu = x;
            sigma = y / 3.0
            distr = distr + y * gaussian(np.linspace(imin - 1, imax + 1, discrete_num), mu, sigma)
        return distr

    def persistence_vector(self, dgm, discete_num=500, debug_flag=False):
        ## here filtration only takes sub or super
        result = self.data_interface(dgm, dynamic_range_flag=self.dynamic_range_flag)
        data = result['data']
        imax = result['max']
        imin = result['min']
        if debug_flag: print(imax, imin)
        data = self.rotate_data(data, super_check=False)
        vector = self.draw_data(data, imax, imin, discrete_num=discete_num)
        vector = np.array(vector).reshape(1, len(vector))
        return vector

    def persistence_vectors(self, dgms, debug='off', axis=1):
        start = time.time()
        n1 = len(dgms)
        n2 = np.shape(self.persistence_vector(dgms[0]))[1]
        X = np.zeros((n1, n2))
        X_list = Parallel(n_jobs=-1)(delayed(unwrap_pdvector)(self, dgms[i]) for i in range(len(dgms)))
        for i in range(n1): X[i] = X_list[i]
        if debug == 'on': print('persistence_vectores takes %s' % (time.time() - start))
        X = normalize(X, norm='l2', axis=axis, copy=True)
        return X

def dgms2feature(dgms, vectype='pi', graphs = None, **params):
    if vectype == 'pi':
        pi = pipl(dgms, type='pi')
        vecs = pi.dgms_vecs(**params)
    elif vectype == 'pl':
        pl = pipl(dgms, type='pl')
        vecs = pl.dgms_vecs(**params)
    elif vectype == 'pvector':
        pdv = pdvector()
        vecs = pdv.persistence_vectors(dgms)
    elif vectype =='bl1':
        diags = dgms2diags(dgms)
        vecs = bl1(diags)
    elif vectype == 'bl0':
        assert graphs is not None
        n = len(graphs)
        vecs = np.zeros((n, 5))
        for i in range(n):
            fval = nx.get_node_attributes(graphs[i][0], params['filtration']).values()
            vecs[i] = stat(fval)

    else:
        raise Exception('vec type error')
    print np.shape(vecs)
    return vecs

def merge_dgms(subdgms, superdgms, vectype='pi', *dgms, **params):
    # merge feature vector of sub_dgms and super_dgms
    # TODO: handle both 0 homology and 1 homology
    sub_vec = dgms2feature(subdgms, vectype=vectype, **params)
    super_vec = dgms2feature(superdgms, vectype=vectype, **params)
    return np.concatenate((sub_vec, super_vec), axis=1)

class classifier():
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.svm_train_stat = None
        self.rf_stat = None
        self.summary = {}
        self.method = kwargs['method']
        # self.direct = kwargs[''] # TODO more generic here

    def rf(self):
        from .svm import rfclf
        if self.method !='bl0' and self.method!='bl1':
            s1, s2, t = rfclf(self.x, self.y, m_f=40, multi_cv_flag=False)
            self.rf_stat = {'s1': s1, 's2': s2, 'time': t}
        else:
            self.rf_stat = {'s1': -1, 's2': -1, 'time': -1} # for baseline, we only have 5 dim feature, but max_feature is 40

    def svm(self):
        from .svm import clf_search_offprint, evaluate_best_estimator
        self.svm_train_stat = clf_search_offprint(self.x, self.y)
        (eval_mean, eval_std) = evaluate_best_estimator(self.svm_train_stat, self.x, self.y, print_flag='off')
        self.svm_eval_stat = {'mean': eval_mean, 'std': eval_std}

    def clf_summary(self, print_flag = False):
        if print_flag:
            if self.svm_train_stat is None:
                print('have not train svm yet')
            else:
                print ('svm train result: %s' %self.svm_train_stat)
                print ('svm eval result: %s' % self.svm_eval_stat)

            if self.rf_stat is None:
                print('have not train random forest yet')
            else:
                print ('rf eval result: %s' % self.rf_stat)

        self.summary['svm_train'] = self.svm_train_stat
        self.summary['svm_eval'] = self.svm_eval_stat
        self.summary['rf_test'] = self.rf_stat
        return self.summary

    # def save_xy(self, x, y):
    #     np.save(self.direct + self.suffix + 'kernel', kernel)




if __name__ == '__main__':
    dgm = d.Diagram([(2,3), (3,4)])
    dgms = [dgm] * 100
    labels = np.array([1, -1]*50)

    params = {'bandwidth': 1.0, 'weight': (1, 1), 'im_range': [0, 1, 0, 1], 'resolution': [25, 25]}
    image = dgms2feature(dgms, vectype='pi', **params)
    images = merge_dgms(dgms, dgms, vectype='pi', **params)
    print (np.shape(image), np.shape(images))

    # c = classifier(images, labels)
    # c.svm()
    # print c.stat()
    # sys.exit()

    params = {'num_landscapes': 5, 'resolution': 100}
    landscape = dgms2feature(dgms, vectype='pl', **params)
    landscapes = merge_dgms(dgms, dgms, vectype='pl', **params)
    print (np.shape(landscape), np.shape(landscapes))

    pd_vector = dgms2feature(dgms, vectype = 'pvector')
    pd_vectors = merge_dgms(dgms, dgms, vectype = 'pvector')
    print (np.shape(pd_vector), np.shape(pd_vectors))
    sys.exit()


