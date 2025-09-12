import abc
import logging
from typing import Callable

import numpy as np

from src.typing import Array1D, Array2D

logger = logging.getLogger(__name__)


class Decomposition(abc.ABC):
    a: Array2D

    @abc.abstractmethod
    def solve(self, b: Array1D) -> Array1D:
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

    def refine(self, b: Array1D, x0: Array1D, tol: float = 1e-5, max_iter: int = 500, new_precision=None):
        '''
        Faz o refinamento do sistema de forma iterativa.
            - b: Valor na equação Ax = b
            - x0: Solução inicial
            - tol (float): tolerância máxima do erro
            - max_iter (int): máximo de iterações permitidas caso a o erro alvo não seja atingido
        '''
        if new_precision is not None: self.precision = new_precision
        x = x0.copy()
        i = 0
        while i < max_iter:
            r = b - self.a @ x
            c = self.solve(r)
            err = norm_inf(c) / norm_inf(x)
            x += c
            if err < tol: break
            i += 1
        return x


def norm_p(x: Array1D, p: int):
    return (sum([abs(xi)**p for xi in x]))**(1/p)


def norm_inf(x: Array1D):
    return max([abs(xi) for xi in x])


def matrix_norm_1(A: Array2D):
    return max([norm_p(A[i, :], 1) for i in range(A.shape[0])])


def matrix_norm_inf(A: Array2D):
    return max([norm_p(A[:, i], 1) for i in range(A.shape[1])])


def matrix_norm_frobenius(A: Array2D):
    acc = 0
    for row in A:
        for x in row:
            acc += x * x
    return acc**(1/2)


def condition_number(A: Array2D, A_inv: Array2D, norm: Callable[[Array2D], float]):
    return norm(A) * norm(A_inv)


def square(a: Array2D):
    '''Testa se um array é matriz quadrada.'''
    return a.ndim == 2 and a.shape[0] == a.shape[1]


def simmetrical(a: Array2D):
    '''Testa se uma matriz é simétrica fazendo sua transposição.'''
    return (a == a.T).all()


def is_lower_trig(A: Array2D):
    '''Testa se uma matriz é triangular inferior.
    Complexidade: O(n²)'''
    A = np.array(A)
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            if A[i, j] != 0:
                return False
    return True


def is_upper_trig(A: Array2D):
    '''Testa se uma matriz é triangular superior.
    Complexidade: O(n²)'''
    A = np.array(A)
    for i in range(A.shape[0]):
        for j in range(i):
            if A[i, j] != 0:
                return False
    return True


def successive_substitutions(A: Array2D, b: Array1D, diag: bool = True):
    '''
    Faz substituições sucessivas em uma matriz escalonada triangular inferior.
    A: matriz triangular inferior
    b: vetor de coeficientes independentes

    Complexidade: O(n²)'''
    A, b = np.array(A), np.array(b)
    b = b.astype(float)
    x = np.zeros_like(b)
    for i in range(A.shape[0]):
        x[i] = (b[i] - (x[:i] @ A[i, :i]))
        if diag:
            x[i] /= A[i, i]
    return x


def retroactive_substitutions(a: Array2D, b: Array1D, diag: bool = True):
    '''
    Faz substituições retroativas em uma matriz escalonada triangular superior.
    a: matriz triangula superiorr
    b: vetor de coeficientes independentes

    Complexidade: O(n²)'''

    a, b = np.array(a), np.array(b)
    b = b.astype(float)
    x = np.zeros_like(b)
    for i in range(a.shape[0] - 1, -1, -1):
        x[i] = (b[i] - (x[i:] @ a[i, i:]))
        if diag:
            x[i] /= a[i, i]
    return x
