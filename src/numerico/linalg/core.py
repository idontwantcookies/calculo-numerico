import pdb
import numpy as np

from .lu import LUDecomp as _LU
from .cholesky import Cholesky as _Cholesky


def square(a):
    return a.shape[0] == a.shape[1]

def simmetrical(a):
    return (a == a.T).all()

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
            a.diagonal().prod()
        elif simmetrical(a):
            l = cholesky(a)
            return l.diagonal().prod()**2
        else:
            dec = _LU()
            l, u, p, sign = dec(a)
            return sign * u.diagonal().prod()
    except ZeroDivisionError:
        return 0

def decomp(a):
    if not square(a): raise ValueError('a must be a square matrix.')
    if simmetrical(a):
        l = cholesky(a)
        u = l.T
        p = np.identity(len(a))
        sign = +1
    else:
        l, u, p, sign = lu(a)
    return l, u, p, sign

def quicksolve(l, u, b):
    if not u.shape == l.shape:
        raise ValueError('L and U must have the same shape.')
    y = successive_substitutions(l, b)
    x = retroactive_substitutions(u, y)
    return x

def solve(a, b):
    l, u, p, sign = decomp(a)
    return quicksolve(l, u, p@b)

def inv(a):
    try:
        l, u, p, sign = decomp(a)
    except ZeroDivisionError:
        raise ZeroDivisionError('Can\'t invert a singular matrix (det = 0).')
    n = len(a)
    e = p @ np.identity(n)
    out = []
    for i in range(n):
        out.append(quicksolve(l, u, e[:, i]))
    return np.array(out).T
