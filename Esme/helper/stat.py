import numpy as np
def statfeat(lis):
    ave_ = np.mean(lis)
    median_ = np.median(lis)
    max_ = np.max(lis)
    min_ = np.min(lis)
    std_ = np.std(lis)
    return np.array((ave_, min_, max_, median_, std_))
