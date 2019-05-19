import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix

from Esme.helper.format import precision_format
from Esme.dgms.format import dgm2diag
from Esme.ml.rf import rfclf

class classifier():
    def __init__(self, x, y, method='svm', n_cv=5, **kwargs):
        """
        classify.
        All hyperparameters are taken care of
        :param x: feature of shape (n_data, dim)
        :param y: label (n_data,)
        :param method: svm, rf
        :param kwargs: pass precomputed kernel as 'kernel'
        """
        self.x = x
        self.y = y
        self.rf_stat = None
        self.summary = {}
        self.n_cv = n_cv
        self.method = method
        if 'kernel' in kwargs.keys(): self.kernel = kwargs['kernel']
        self.stat = {'train': None, 'test': None}
        # self.direct = kwargs[''] # TODO more generic here

    def rf(self):
        if self.method !='bl0' and self.method!='bl1':
            s1, s2, t = rfclf(self.x, self.y, m_f=40, multi_cv_flag=False)
            self.rf_stat = {'s1': s1, 's2': s2, 'time': t}
        else:
            self.rf_stat = {'s1': -1, 's2': -1, 'time': -1} # for baseline, we only have 5 dim feature, but max_feature is 40

    def svm(self, n_splits = 10):
        # linear/Gaussian kernel

        self.stat['train'] = train_svm(self.x, self.y)
        eval_mean, eval_std, n_cv = evaluate_best_estimator(self.stat['train'], self.x, self.y, print_flag='off', n_splits = n_splits, n_cv=self.n_cv)
        self.stat['test'] = {'mean': eval_mean, 'std': eval_std, 'n_cv': n_cv}

    def svm_kernel_(self, n_splits = 10):
        # precomputed kernel

        self.stat['train'] = train_svm(self.x, self.y, kernel_flag=True, kernel=self.kernel, print_flag='off')
        eval_mean, eval_std, n_cv = evaluate_best_estimator(self.stat['train'], self.x, self.y, print_flag='off', kernel=self.kernel, n_splits=n_splits, n_cv = self.n_cv)
        self.stat['test'] = {'mean': eval_mean, 'std': eval_std, 'n_cv': n_cv}

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

def dgms2swdgm(dgms):
    swdgms=[]
    for dgm in dgms:
        diag = dgm2diag(dgm)
        swdgms += [np.array(diag)]
    return swdgms

def train_svm(x, y, random_state=2, print_flag='off', nonlinear_flag = True, kernel_flag=False, kernel=np.zeros((1, 1))):
    """
    :param x: feature
    :param y: label
    :param random_state: random seed for 10 cv
    :param print_flag: 'on'/'off' for debug
    :param nonlinear_flag: linear
    :param kernel_flag: True if use precomputed kernel
    :param kernel: precomputed kernel. No need to pass if use gaussian/linear kernel
    :return: best parameters
    """
    tuned_params = [{'kernel': ['linear'], 'C': [0.01, 1, 10, 100, 1000]}]
    if nonlinear_flag:
        tuned_params += [{'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10, 100], 'C': [0.1, 1, 10, 100,1000]}]

    for score in ['accuracy']:
        x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, range(len(y)), test_size=0.1, random_state=random_state)

        if not kernel_flag: # not precomputed kernel
            clf = GridSearchCV(svm.SVC(), tuned_params, cv=10, scoring='%s' % score, n_jobs=-1, verbose=0)
            clf.fit(x_train, y_train)
        else:
            clf = GridSearchCV(svm.SVC(kernel='precomputed'),
                               [{'C': [0.01, 0.1, 1, 10, 100, 1000]}],
                               cv=10, scoring='%s' % score, n_jobs=-1, verbose=0)
            kernel_train = kernel[np.ix_(indices_train, indices_train)]
            clf.fit(kernel_train, y_train)
            assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
            kernel_test = kernel[np.ix_(indices_test, indices_train)]

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        if print_flag == 'on':
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print("Detailed classification report:\n")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.\n")

            if kernel_flag == False:
                y_true, y_pred = y_test, clf.predict(x_test)
            else:
                y_true, y_pred = y_test, clf.predict(kernel_test)
                print('Able to execute kernel grid search')
            print(accuracy_score(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            print(confusion_matrix(y_true, y_pred))

        if 'kernel' not in clf.best_params_:
            clf.best_params_['kernel'] = 'precomputed'
        return {'param': clf.best_params_, 'score': round(clf.best_score_ * 1000)/10.0}

def evaluate_tda_kernel(tda_kernel, Y, best_result_so_far, print_flag='off'):
    """
    TODO: figure this out
    :param tda_kernel:
    :param Y:
    :param best_result_so_far:
    :param print_flag:
    :return:
    """

    t1 = time.time()
    n = np.shape(tda_kernel)[0]
    grid_search_re = train_svm(np.zeros((n, 23)), Y, print_flag=print_flag, kernel=tda_kernel,
                               kernel_flag=True, nonlinear_flag=False)  # X is dummy here
    if grid_search_re['score'] < best_result_so_far[0]-4:
        print('Saved one unnecessary evaluation of bad kernel')
        return (0,0,{},0)

    cv_score = []
    for seed in range(5):
        clf = svm.SVC(kernel='precomputed', C=grid_search_re['param']['C'])
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, tda_kernel, Y, cv=k_fold, scoring='accuracy', n_jobs=-1)
        cv_score.append(scores.mean())

    cv_score = np.array(cv_score)
    t2 = time.time()
    svm_time = precision_format(t2 - t1, 1)
    return (precision_format(100 * cv_score.mean(), 1),
            precision_format(100 * cv_score.std(), 1),
            grid_search_re, svm_time)

def evaluate_best_estimator(grid_search_re, x, y, print_flag='off', kernel=None, n_splits = 10, n_cv=5):
    """
    :param grid_search_re: grid search result(dict)
    :param x: feat
    :param y: label
    :param print_flag: on/off
    :param kernel:
    :param n_splits:
    :param n_cv: number of cv(5/10) for evaluation
    :return:
    """
    if print_flag=='on': print('Start evaluating the best estimator')
    param = grid_search_re['param']
    assert param['kernel'] in ['linear', 'rbf', 'precomputed']
    assert isinstance(param, dict)

    # set up clf
    if len(param) == 3:
        clf = svm.SVC(kernel='rbf', C=param['C'], gamma = param['gamma'])
    elif (len(param) == 2) and (param['kernel'] == 'linear'):
        clf = svm.SVC(kernel='linear', C = param['C'])
    else:
        clf = svm.SVC(kernel='precomputed', C=param['C'])

    # evaluation
    t0 = time.time()
    cv_score, n_cv = [], n_cv
    for i in range(n_cv):
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
        if param['kernel']!= 'precomputed':
            scores = cross_val_score(clf, x, y, cv=k_fold, scoring='accuracy', n_jobs=-1)
        else:
            scores = cross_val_score(clf, kernel, y, cv=k_fold, scoring='accuracy', n_jobs=-1)

        if print_flag == 'on': print(scores)
        cv_score.append(scores.mean())
    cv_score = np.array(cv_score)

    if print_flag=='on':
        print(cv_score)

    print('Evaluation takes %0.3f. '
          'After averageing %0.1f cross validations, the mean accuracy is %0.3f, the std is %0.3f\n'
          %(time.time()-t0, n_cv, cv_score.mean(), cv_score.std()))
    return cv_score.mean(), cv_score.std(), n_cv

if __name__=='__main__':
    import sklearn.datasets as datasets
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    x = iris.data[:, :2]
    y = iris.target
    clf = classifier(x, y, method='svm', n_cv=1)
    clf.svm()

    # train stat: {'param': {'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}, 'score': 80.0}
    # test stat: {'mean': 0.8226666666666667, 'std': 0.007999999999999896}
    print(clf.stat)