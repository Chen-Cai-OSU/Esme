import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(2, 0.4)
neigh.fit(samples)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time


def knn(x, y, k = 3, n_iter=2, print_flag=False):
    t0 = time.time()
    iris = datasets.load_iris()
    # x = iris.data[:, :2]
    # y = iris.target
    clf = KNeighborsClassifier(n_neighbors=k)
    res = []
    for i in range(n_iter):
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        score1 = cross_val_score(clf, x, y, cv=k_fold, n_jobs=-1)
        res.append(score1)

    mean_ = np.mean(list(map(np.mean, res)))
    std_ = np.mean(list(map(np.std, res)))
    if print_flag:
        print('Evaluation takes %0.3f. After averageing %0.1f cross validations, '
              'the mean accuracy is %0.3f, the std is %0.3f\n' % ((time.time() - t0, n_iter, mean_, std_)))

    return mean_, std_

if __name__ == '__main__':
    print(__doc__)
    knn(n_iter=10, print_flag=True)

