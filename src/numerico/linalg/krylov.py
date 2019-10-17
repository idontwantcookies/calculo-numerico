import numpy as np

from .lu import LU


def krylov_poly(a):
    a = np.array(a)
    n = len(a)
    y = np.array([1] + [0] * (n - 1))
    Y = []
    for i in range(n):
        Y.insert(0, y)
        y = a @ y
    y_n = y.copy()
    Y = np.array(Y).T
    lu = LU(Y)
    coefs = list(lu.solve(-y_n))
    coefs.insert(0, 1)
    return np.poly1d(coefs)
