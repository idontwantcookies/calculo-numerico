import abc

import numpy as np

def get_col(a, col):
    out = []
    for i in range(len(a)):
        out.append(a[i][col])
    return out

def argmax(array):
    return array.index(max(array, key=lambda x: abs(x)))

def get_pivot(array, p):
    col = get_col(array, p)[p:]
    return argmax(col) + p

def zeros(m, n):
    out = []
    for i in range(m):
        out.append([])
        for j in range(n):
            out[i].append(0)
    return out

def identity(n):
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1
    return out

class Decomposition(abc.ABC):
    @abc.abstractmethod
    def solve(self, b):
        pass

    @abc.abstractmethod
    def det(self):
        pass

    @abc.abstractmethod
    def inv(self):
        pass

    @abc.abstractmethod
    def _execute(self):
        pass

    def refine(self, b, x0, tol=1e-5, max_iter=500, new_precision=None):
        if new_precision is not None: self.precision = new_precision
        x = x0.copy()
        i = 0
        while i < max_iter:
            r = b - self.a @ x
            c = np.array(self.solve(r))
            err = max(abs(c / x))
            x += c
            if err < tol: break
            i += 1
        return x

def successive_substitutions(a, b, diag=True):
    '''
    Faz substituições sucessivas em uma matriz escalonada triangular inferior.
    a: matriz triangular inferior
    b: vetor de coeficientes independentes

    Complexidade: ~n²'''
    n = len(a)
    x = [0] * n
    for i in range(n):
        x[i] = b[i]
        for k in range(i):
            x[i] -= x[k] * a[i][k]
        if diag:
            x[i] /= a[i][i]
    return x

def retroactive_substitutions(a, b, diag=True):
    '''
    Faz substituições retroativas em uma matriz escalonada triangular superior.
    a: matriz triangula superior
    b: vetor de coeficientes independentes

    Complexidade: ~n²'''

    n = len(a)
    x = [0] * n
    for i in range(n-1,-1,-1):
        x[i] = b[i]
        for k in range(i + 1, n):
            x[i] -= x[k] * a[i][k]
        if diag:
            x[i] /= a[i][i]
    return x

def solve_diag(a, b):
    n = len(a)
    x = [0] * n
    for i in range(n):
        x[i] = b[i] / a[i][i]
    return x

def lu_solve(a, b, e=None):
    n = len(b)
    if e is None:
        e = identity(n)
    y = zeros(n)
    for i in range(n):
        y[i] = b[e[i].index(1)]
    t = successive_substitutions(a, y, diag=False)
    x = retroactive_substitutions(a, t, diag=True)
    return x

def swap_rows(a, row1, row2):
    aux = a[row1].copy()
    a[row1] = a[row2]
    a[row2] = aux

def gauss(a, b, pivoting=True, debug=False, inplace=False):
    if not inplace: a, b = a.copy(), b.copy()
    n, det = len(a), 1
    for p in range(n):  # linha pivotal
        if pivoting:
            new_p = get_pivot(a, p)
            swap_rows(a, p, new_p)
            swap_rows(a, p, new_p)
            if new_p != p: det *= -1
        if debug: print(a[p:]); print('-' * 80)
        det *= a[p][p]
        for i in range(p + 1, n):
            m = a[i][p] / a[p][p]
            b[i] -= m * b[p]
            for j in range(p, n):
                a[i][j] -= m * a[p][j]
    x = retroactive_substitutions(a, b)
    return x, det

def lu(a, pivoting=True, debug=False, inplace=False):
    if not inplace: a = a.copy()
    n, det = len(a), 1
    e = type(a)(identity(n))    # keeps track of swaps on identity matrix
    for p in range(n):
        if pivoting:
            new_p = get_pivot(a, p)
            swap_rows(a, p, new_p)
            swap_rows(e, p, new_p)
            if new_p != p: det *= -1
        if debug: print(a[p:]); print('-' * 80)
        det *= a[p][p]
        for i in range(p + 1, n):
            m = a[i][p] / a[p][p]
            a[i][p] = m
            for j in range(p + 1, n):
                a[i][j] -= m * a[p][j]
    return a, det, e
