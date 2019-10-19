import abc

import numpy as np


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
            c = self.solve(r)
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
    a, b = np.array(a), np.array(b)
    b = b.astype(a.dtype)
    x = np.zeros_like(b)
    for i in range(a.shape[0]):
        x[i] = b[i]
        print(f'x{i} = ({b[i]}', end='')
        for k in range(i):
            x[i] -= x[k] * a[i, k]
            print(' - ', f'({x[k]})({a[i,k]})', end='')
        print(')', end='')
        if diag:
            x[i] /= a[i, i]
            print(f'/ {a[i,i]}', end='')
        print(f' = {x[i]}')
    return x

def retroactive_substitutions(a, b, diag=True):
    '''
    Faz substituições retroativas em uma matriz escalonada triangular superior.
    a: matriz triangula superior
    b: vetor de coeficientes independentes

    Complexidade: ~n²'''

    a, b = np.array(a), np.array(b)
    b = b.astype(a.dtype)
    x = np.zeros_like(b)
    for i in range(a.shape[0] - 1, -1, -1):
        x[i] = (b[i] - x[i:] @ a[i, i:])
        if diag:
            x[i] /= a[i, i]
    return x
