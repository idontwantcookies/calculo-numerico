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



def square(a):
    '''Testa se um array é matriz quadrada.'''
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def simmetrical(a):
    '''Testa se uma matriz é simétrica fazendo sua transposição.'''
    return (a == a.T).all()

def is_lower_trig(a):
    '''Testa se uma matriz é triangular inferior.
    Complexidade: ~n²/2 comparações (pior caso)'''
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[1]):
            if a[i,j] != 0: return False
    return True

def is_upper_trig(a):
    '''Testa se uma matriz é triangular superior.
    Complexidade: ~n²/2 comparações (pior caso)'''
    for i in range(a.shape[0]):
        for j in range(i):
            if a[i,j] != 0: return False
    return True

def successive_substitutions(a, b, diag=True):
    '''
    Faz substituições sucessivas em uma matriz escalonada triangular inferior.
    a: matriz triangular inferior
    b: vetor de coeficientes independentes

    Complexidade: ~n²'''
    b = b.copy().astype(a.dtype)
    x = np.zeros_like(b)
    for i in range(a.shape[0]):
        x[i] = (b[i] - (x[:i] @ a[i, :i]))
        if diag:
            x[i] /= a[i, i]
    return x

def retroactive_substitutions(a, b, diag=True):
    '''
    Faz substituições retroativas em uma matriz escalonada triangular superior.
    a: matriz triangula superiorr
    b: vetor de coeficientes independentes

    Complexidade: ~n²'''

    b = b.copy().astype(a.dtype)
    x = np.zeros_like(b)
    for i in range(a.shape[0] - 1, -1, -1):
        x[i] = (b[i] - (x[i:] @ a[i, i:]))
        if diag:
            x[i] /= a[i, i]
    return x
