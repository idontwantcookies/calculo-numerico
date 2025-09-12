import logging

import numpy as np
from src.typing import Array1D, Array2D

from src.linalg.core import retroactive_substitutions


logger = logging.getLogger(__name__)


def swap(m, i, j):
    aux = m[i].copy()
    m[i] = m[j].copy()
    m[j] = aux
    return m


def swap_pivot(m, pivot_index):
    col = m[pivot_index:, pivot_index]
    new_index = col.argmax() + pivot_index
    if pivot_index != new_index:
        swap(m, pivot_index, new_index)
        return -1
    else:
        return 1


def gauss(a: Array2D, b: Array1D, pivoting=True, precision=None):
    a, b = a.copy(), b.copy()
    det = 1
    n = len(a)
    M = np.zeros((n, n+1))
    M[:, :-1] = a.copy().astype(float)
    M[:, -1] = b.copy().astype(float)
    for pivot in range(n):
        if pivoting:
            det *= swap_pivot(M, pivot)
        logger.info(f'\n{M}')
        logger.info(f'\n{"-" * 80}')
        det *= M[pivot, pivot]
        m = - 1 / M[pivot, pivot]
        for i in range(pivot + 1, n):
            M[i] += m * M[i, pivot] * M[pivot]
        if precision is not None:
            M = M.round(precision)
    x = retroactive_substitutions(M[:, :-1], M[:, -1])
    return x, det
