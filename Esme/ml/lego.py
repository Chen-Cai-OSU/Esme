# some simple ml functions
import numpy as np


def onehot(n):
    y = np.zeros((2 * n, 2))
    for i in range(n):
        y[i][0] = -1
    for i in range(n, 2 * n):
        y[i][1] = 1
    return y


def hinge_loss(y, y_hat):
    assert ((y == 1) or (y == -1))
    if y == 1:
        return max(0, 1 - y_hat)
    elif y == -1:
        return max(0, 1 + y_hat)


def hinge_gradient(y, y_hat):
    # scalar version. The gradient of hinge loss(1-y_hat*y)+
    assert (y == 1) or (y == -1)
    if y == 1:
        if y_hat >= 1:
            return 0
        elif y_hat < 1:
            return -1
    elif y == -1:
        if y_hat >= -1:
            return 1
        elif y_hat < -1:
            return 0
