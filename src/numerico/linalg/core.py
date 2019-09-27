import numpy as np

from .lu import LUDecomp as _LU
from .cholesky import Cholesky as _Cholesky


def main_diag_product(a):
    prod = 1
    for i in range(a.shape[0]):
        prod *= a[i, i]
    return prod

def lu(a, *args, **kwargs):
    dec = _LU(*args, **kwargs)
    return dec(a)

def cholesky(a, *args, **kwargs):
    dec = _Cholesky(*args, **kwargs)
    return dec(a)

def is_lower_trig(a):
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[1]):
            if a[i,j] != 0: return False
    return True

def is_upper_trig(a):
    for i in range(a.shape[0]):
        for j in range(i):
            if a[i,j] != 0: return False
    return True

def successive_substitutions(a, b):
    if not is_lower_trig(a):
        raise ValueError('a must be a lower trig matrix.')
    b = b.copy()
    x = np.zeros_like(b)
    for i in range(a.shape[0]):
        x[i] = (b[i] - (x[:i] * a[i, :i]).sum()) / a[i, i]
    return x

def retroactive_substitutions(a, b):
    if not is_upper_trig(a):
        raise ValueError('a must be an upper trig matrix.')
    b = b.copy()
    x = np.zeros_like(b)
    for i in range(a.shape[0] - 1, -1, -1):
        x[i] = (b[i] - (x[i:] * a[i, i:]).sum()) / a[i, i]
    return x

def det(a):
    try:
        if is_lower_trig(a) or is_upper_trig(a):
            return main_diag_product(a)
        elif (a == a.T).all():
            l = cholesky(a)
            return main_diag_product(l)**2
        else:
            dec = _LU()
            l, u, p, sign = dec(a)
            return sign * main_diag_product(u)
    except ZeroDivisionError:
        return 0

def quicksolve(l, u, b):
    if not u.shape == l.shape:
        raise ValueError('L and U must have the same shape.')
    y = successive_substitutions(l, b)
    x = retroactive_substitutions(u, y)
    return x

def solve(a, b):
    if (a == a.T).all():
        l = cholesky(a)
        return quicksolve(l, l.T, b)
    else:
        l, u, p, sign = lu(a)
        return quicksolve(l, u, p@b)
