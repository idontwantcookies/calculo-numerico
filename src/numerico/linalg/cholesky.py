import numpy as np


class Cholesky:
    '''
    Inicializa uma classe Cholesky que pode ser usada em decomposição de uma 
    matriz quadrada simétrica.

    Complexidade: ~n³/3

    Uso:
    >>> import numpy as np
    >>> A = np.array([[1,2],[2,5]])
    >>> dec = Cholesky()
    >>> L = dec(A)
    >>> print(L)
    [[1. 0.]
     [2. 1.]]
    
    A matriz L pode ser de números complexos caso A não seja definida positiva.
    '''


    def __init__(self, precision=None):
        '''
        Parâmetros: 
        precision: int (padrão None)
            Use se quiser arredondar para x casas decimais ao final de cada 
            iteração (para comparar com exercícios durante estudos).
        '''

        self.precision = precision

    def _setUp(self, a):
        self.a = a
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
            self.L = self.L._round(self.precision)

    def __call__(self, a):
        '''
        Parâmetros:
        a: np.array bidimensional e quadrado
            A matriz que se deseja decompor.
        ---------
        Retorno:
        L: np.array bidimensional e quadrado
            Matriz triangular inferior tal que L @ L.T == A.
        '''

        self._setUp(a)
        for j in range(self.n):
            self._set_diagonal_element(j)
            for i in range(j+1, self.n):
                self._set_non_diagonal_element(i, j)
            self._round()
        if (self.L.real == self.L).all(): self.L = self.L.real
        return self.L
