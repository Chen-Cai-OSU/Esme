import dionysus as d
import numpy as np

from joblib import delayed, Parallel


class test_dionysus():
    pass

    def test_dionysus(self):
        dgm = d.Diagram([(1, 2), (3, 4)])
        for i in range(100):
            print((dgm[i]))
        for p in dgm:
            print(p)

    def computePD(i):
        # print('Process %s' % i)
        import dionysus as d
        import numpy as np
        # np.random.seed(42)
        f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
        m1 = d.homology_persistence(f1)
        dgms1 = d.init_diagrams(m1, f1)
        # return [(p.birth, p.death) for p in dgms1[1]]  # advice from Dmitriy
        return dgms1[1]

    computePD = staticmethod(computePD)


def bad_example():
    import dionysus as d
    dgm1 = d.Diagram([(1, 2.07464)])
    dgm1 = d.Diagram([(1, 2.04287)])
    dgm2 = d.Diagram([(1, 1.68001), (1, 1.68001), (1, 1.68001)])  # this one doesn't work
    dgm2 = d.Diagram([(1, 1.71035)])
    # dgm2 = d.Diagram([(1,1.68), (1,1.68), (1,1.68)]) # But this one works
    print((d.bottleneck_distance(dgm1, dgm2)))
    print((d.bottleneck_distance_with_edge(dgm1, dgm2)))

def computePD(i):
    # print('Process %s' % i)
    # np.random.seed(42)
    f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
    m1 = d.homology_persistence(f1)
    dgms1 = d.init_diagrams(m1, f1)
    # return [(p.birth, p.death) for p in dgms1[1]]  # advice from Dmitriy
    return dgms1[1]


def test_get_dgms(n_jobs=1):
    return Parallel(n_jobs=-1)(delayed(computePD)(i) for i in range(70))

def test_dionysus_modification():
    simplices = [([2], 4), ([1, 2], 5), ([0, 2], 6),
                 ([0], 1), ([1], 2), ([0, 1], 3)]
    f = d.Filtration()
    for vertices, time in simplices:
        f.append(d.Simplex(vertices, time))

    def compare(s1, s2, sub_flag=True):
        if sub_flag == True:
            if s1.dimension() > s2.dimension():
                return 1
            elif s1.dimension() < s2.dimension():
                return -1
            else:
                return cmp(s1.data, s2.data)
        elif sub_flag == False:
            return -compare(s1, s2, sub_flag=True)

    f.sort(cmp=compare)
    for s in f:
        print(s)
