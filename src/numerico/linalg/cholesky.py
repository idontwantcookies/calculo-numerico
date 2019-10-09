import numpy as np

from .core import Decomposition, successive_substitutions, \
                  retroactive_substitutions


class Cholesky(Decomposition):
    '''
    Inicializa uma classe Cholesky que pode ser usada em decomposição de uma 
    matriz quadrada simétrica.

    Complexidade: ~n³/3

    Uso:
    >>> import numpy as np
    >>> A = np.array([[1,2],[2,5]])
    >>> dec = Cholesky(A)
    >>> print(dec.L)
    [[1. 0.]
     [2. 1.]]
    
    A matriz L pode ser de números complexos caso A não seja definida positiva.
    '''


    def __init__(self, a, precision=None):
        '''
        Parâmetros:
        a: np.array 2d
            Matriz simétrica a ser decomposta
        precision: int (padrão None)
            Use se quiser arredondar para x casas decimais ao final de cada 
            iteração (para comparar com exercícios durante estudos).
        '''

        self.a = a
        self.precision = precision
        self._setUp()
        self._execute()


    def _setUp(self):
        self.n = self.a.shape[0]
        self.L = np.zeros_like(self.a).astype(complex)

    def _set_diagonal_element(self, j):
        part_sum = self.a[j, j] - sum(self.L[j]**2)
        if part_sum == 0:
            raise ZeroDivisionError('Can\'t decompose singular matrix (det = 0).')
        self.L[j, j] = part_sum ** 0.5

    def _set_non_diagonal_element(self, i, j):
        pivot = self.L[j, j]
        row = self.L[i, :i]
        col = self.L[j, :i]
        part_sum = row @ col
        self.L[i, j] = (self.a[i, j] - part_sum) / pivot

    def _round(self):
        if self.precision is not None:
            self.L = self.L.round(self.precision)

    def _execute(self):
        for j in range(self.n):
            self._set_diagonal_element(j)
            for i in range(j+1, self.n):
                self._set_non_diagonal_element(i, j)
            self._round()
        if (self.L.real == self.L).all(): self.L = self.L.real

    def solve(self, b):
        t = successive_substitutions(self.L, b)
        x = retroactive_substitutions(self.L.T, t)
        return x

    def inv(self):
        e = np.identity(self.n)
        out = np.zeros_like(self.a).astype(float)
        for i in range(self.n):
            out[:,i] = self.solve(e[:,i])
        return out

    @property
    def det(self):
        if not hasattr(self, '_det'):
            prod = 1
            for i in range(self.n):
                prod *= self.L[i,i]
            self._det = prod**2
        return self._det
