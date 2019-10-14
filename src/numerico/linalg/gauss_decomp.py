import pdb
import numpy as np

from .core import retroactive_substitutions


def swap(m, i, j):
    aux = m[i].copy()
    m[i] = m[j].copy()
    m[j] = aux
    return m

def swap_pivot(m, pivot_index):
    col = m[pivot_index:, pivot_index]
    new_index = col.argmax() + pivot_index
    if pivot_index  != new_index:
        swap(m, pivot_index, new_index)
        return -1
    else:
        return 1

def gauss(a, b, pivoting=True, debug=False, precision=None):
    det = 1
    n = len(a)
    M = np.zeros((n, n+1))
    M[:,:-1] = a.copy().astype(float)
    M[:,-1] = b.copy().astype(float)
    for pivot in range(n):
        if pivoting: det *= swap_pivot(M, pivot)
        if debug: print(M); print('-' * 80)
        det *= M[pivot, pivot]
        m = - 1 / M[pivot, pivot]
        for i in range(pivot + 1, n):
            M[i] += m * M[i, pivot] * M[pivot]
        if precision is not None: M = M.round(precision)
    x = retroactive_substitutions(M[:,:-1], M[:,-1])
    return x, det
