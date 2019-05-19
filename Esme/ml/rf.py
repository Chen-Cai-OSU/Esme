import time
import numpy as np
from ..helper.format import precision_format
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def rfclf(x, y, m_f = 40, multi_cv_flag=False, print_flag=False):
    if np.shape(x)==(1, 2):
        return

    clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    score1 = np.array([0, 0])
    score1 = cross_val_score(clf1, x, y, cv=10, n_jobs=-1)
    ten_score = []
    if multi_cv_flag==True:
        n_iter = 10
        print ('Using rf 10 cv 10 times')
    else: n_iter=1

    time1 = time.time()
    for seed in range(n_iter):
        clf2 = RandomForestClassifier(n_estimators=40, random_state = seed, max_features=m_f, max_depth=None, min_samples_split=2,n_jobs=-1)
        score2 = cross_val_score(clf2, x, y, cv=10, n_jobs=-1)
        ten_score += list(score2)
    score2 = np.array(ten_score)
    time2 = time.time()
    rf_time = precision_format(time2-time1,1)
    if print_flag:
        print('Try Random Forest, n_estimators=40, max_features=', m_f),
        print('Accuracy is %s'%score2.mean())


    return (score1.mean(), score2.mean(), rf_time/100.0)
