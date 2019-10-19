import abc

import numpy as np

def get_col(a, col):
    out = []
    for i in range(len(a)):
        out.append(a[i][2])
    return out

def argmax(array):
    return array.index(max(array))


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
            x[i] -= x[k] * a[i, k]
        if diag:
            x[i] /= a[i, i]
    return x

def retroactive_substitutions(a, b, diag=True):
    '''
    Faz substituições retroativas em uma matriz escalonada triangular superior.
    a: matriz triangula superior
    b: vetor de coeficientes independentes

    Complexidade: ~n²'''

    n = len(a)
    x = [0] * len(b)
    for i in range(n-1,-1,-1):
        x[i] = b[i]
        for k in range(i + 1, n):
            x[i] -= x[k] * a[i,k]
        if diag:
            x[i] /= a[i, i]
    return x

def gauss(a, b, pivoting=True, debug=False):
    a, b, n, det = a.copy(), b.copy(), len(a), 1
    for p in range(n):  # linha pivotal
        if pivoting:
            new_p = argmax(get_col(a, p))
            a[p], a[new_p] = a[new_p], a[p]
            b[p], b[new_p] = b[new_p], b[p]
            if new_p != p: det *= -1
        if debug: print(a); print('-' * 80)
        det *= a[p, p]
        m = 1 / a[p, p]
        for i in range(p + 1, n):
            for j in range(p + 1, n):
                a[i][j] -= m * a[i, p] * a[p]
    x = retroactive_substitutions(a, b)
    return x, det
