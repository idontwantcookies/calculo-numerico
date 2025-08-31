import numpy as np

from src.linalg.core import Decomposition, retroactive_substitutions, \
    successive_substitutions


class LU(Decomposition):
    '''
    Classe responsável por executar a decomposição LU de uma matriz.

    Uso:
    >>> import numpy as np
    >>> from src.linalg import LU
    >>> A = np.array([[ 1, -3,  2],
                      [-2,  8, -1],
                      [ 4, -6,  5]])
    >>> dec = LU(A)
    >>> print(dec.LU)
    [[ 4.   -6.    5.  ]
     [-0.5   5.    1.5.  ]
     [ 0.25 -0.3   1.2.  ]]
    >>> print(dec.p)
    [[0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]]
    >>> print(dec.det())
    -24.0
    >>> print(dec.inv())
    [[-1.41666667 -0.125       0.54166667]
     [-0.25        0.125       0.125     ]
     [ 0.83333333  0.25       -0.08333333]]
    '''

    def __init__(self, a, pivoting=True, debug=False, precision=None):
        '''
        Parametros:
        a: np.array 2d quadrado
            Matriz a ser decomposta
        pivoting: bool (padrão True)
            Determina se deve usar pivotação parcial.
        debug: bool (padrão False)
            Determina se deve imprimir linhas das etapas da decomposição LU.
        precision: int (padrão None)
            Determina quantas casas decimais usar no arredondamento a cada 
            iteração externa.
        '''

        self.a = np.array(a).astype(float)
        self.pivoting = pivoting
        self.debug = debug
        self.precision = precision
        self._setUp()
        self._execute()

    @property
    def det(self):
        if not hasattr(self, '_det'):
            self._det = self.LU.diagonal().prod() * (-1) ** self.swap_count
        return self._det

    def solve(self, b):
        b = np.array(b)
        t = successive_substitutions(self.LU, self.p @ b, diag=False)
        x = retroactive_substitutions(self.LU, t)
        return x

    def inv(self):
        out = np.zeros_like(self.LU)
        e = np.identity(self.N)
        for i in range(self.N):
            out[:, i] = self.solve(e[:, i])
        return out

    def _setUp(self):
        # Cria as matrizes LU e p.
        self.N = self.a.shape[0]
        self.LU = self.a.copy()
        self.p = np.identity(self.N)

    def _swap_rows(self, matrix, i, j):
        # troca as linhas i e j de uma matriz qualquer.
        aux = matrix[i].copy()
        matrix[i] = matrix[j]
        matrix[j] = aux

    def _max(self, row):
        # retorna o maior valor e o índice desse valor de um np.array 1d.
        result = 0
        index = 0
        for i, x in enumerate(row):
            if abs(x) > abs(result):
                result = x
                index = i
        return index, result

    def _pick_pivot(self, column):
        # escolhe um pivô dada uma coluna. Retorna seu índice e seu valor.
        if self.pivoting:
            pivotal_col = self.LU[column:, column]
            i, pivot = self._max(pivotal_col)
            i += column
        else:
            i = column
            pivot = self.LU[i, i]
            if pivot == 0:
                raise ZeroDivisionError(
                    '0 as a pivot found. Please try setting pivoting=True.')
        return i, pivot

    def _swap(self, i, j):
        # troca as linhas da matriz LU e da matriz p.
        if i != j:
            self.swap_count += 1
            self._swap_rows(self.LU, i, j)
            self._swap_rows(self.p, i, j)

    def _show_steps(self, current_pivot):
        # imprime todas as linhas da matriz LU começando pela linha pivotal atual.
        if self.debug:
            print(self.LU[current_pivot:])
            print('-' * 80)

    def _apply_pivot(self, pivot, cur):
        # modifica in-place uma linha cur somada a uma linha pivotal (pivot) dada.
        p = self.LU[pivot, pivot]
        mult = self.LU[cur, pivot] / p
        self.LU[cur, pivot] = mult
        self.LU[cur, pivot + 1:] -= mult * self.LU[pivot, pivot + 1:]

    def _round(self):
        # arredonda toda a matriz LU
        if self.precision is not None:
            self.LU = self.LU.round(self.precision)

    def _execute(self):
        # executa a decomposição
        self.swap_count = 0
        self._setUp()
        for pivot_line in range(self.N):
            i, pivot = self._pick_pivot(pivot_line)
            if pivot == 0:
                self.LU[i:, i:] = 0
                break
            self._swap(i, pivot_line)
            self._round()
            self._show_steps(pivot_line)
            for cur_line in range(pivot_line + 1, self.N):
                self._apply_pivot(pivot_line, cur_line)
