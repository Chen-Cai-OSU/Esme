import time

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV

from Esme.dgms.format import dgm2diag
from Esme.helper.format import precision_format


# from Esme.permutation tools import precision_format, dgms_summary, add_dgms, dgm_vec, evaluate_best_estimator
# from .tools import precision_format, dgms_summary, add_dgms, dgm_vec, dgm2diag, evaluate_best_estimator # may change back to this

def dgms2swdgm(dgms):
    swdgms=[]
    for dgm in dgms:
        diag = dgm2diag(dgm)
        swdgms += [np.array(diag)]
    return swdgms

def clf_search_offprint(X, Y, random_state=2, print_flag='off', nonlinear_flag = True, kernel_flag=False, kernel=np.zeros((1,1))):
    if nonlinear_flag == True:
        tuned_params = [{'kernel': ['linear'], 'C': [ 0.1, 1, 10, 100, 1000]}, #[ 0.1, 1, 10, 50, 100, 1000]},
                        {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10, 100], 'C': [0.1, 1, 10, 100,1000]}]
    else:
        tuned_params = [{'kernel': ['linear'], 'C': [ 0.01, 1, 10, 100, 1000]}]

    for score in ['accuracy']:
        X_train, X_test, y_train, y_test, indices_train, indices_test \
            = train_test_split(X, Y, range(len(Y)), test_size=0.1, random_state=random_state)

        if kernel_flag == False: # not precomputed kernel
            clf = GridSearchCV(svm.SVC(), tuned_params, cv=10, scoring='%s' % score, n_jobs=-1, verbose=0)
            clf.fit(X_train, y_train)
        else:
            clf = GridSearchCV(svm.SVC(kernel='precomputed'), [{'C': [0.01, 0.1, 1, 10, 100, 1000]}],
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
                y_true, y_pred = y_test, clf.predict(X_test)
            else:
                y_true, y_pred = y_test, clf.predict(kernel_test)
                print('Able to execute kernel grid search')
            print(accuracy_score(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_true, y_pred))

        return {'param': clf.best_params_, 'score': round(clf.best_score_ * 1000)/10.0}

def evaluate_tda_kernel(tda_kernel, Y, best_result_so_far, print_flag='off'):
    t1 = time.time()
    n = np.shape(tda_kernel)[0]
    grid_search_re = clf_search_offprint(np.zeros((n, 23)), Y, print_flag=print_flag, kernel=tda_kernel,
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

def rfclf(X,Y, m_f = 40, multi_cv_flag=False, print_flag=False):
    import time
    import numpy as np
    if np.shape(X)==(1,2):
        return
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    score1 = np.array([0, 0])
    score1 = cross_val_score(clf1, X, Y, cv=10, n_jobs=-1)
    ten_score = []
    if multi_cv_flag==True:
        n_iter = 10
        print ('Using rf 10 cv 10 times')
    else: n_iter=1

    time1 = time.time()
    for seed in range(n_iter):
        clf2 = RandomForestClassifier(n_estimators=40, random_state = seed, max_features=m_f, max_depth=None, min_samples_split=2,n_jobs=-1)
        score2 = cross_val_score(clf2, X, Y, cv=10, n_jobs=-1)
        ten_score += list(score2)
    score2 = np.array(ten_score)
    time2 = time.time()
    rf_time = precision_format(time2-time1,1)
    if print_flag:
        print('Try Random Forest, n_estimators=40, max_features=', m_f),
        print('Accuracy is %s'%score2.mean())
    return (score1.mean(), score2.mean(), rf_time/100.0)
