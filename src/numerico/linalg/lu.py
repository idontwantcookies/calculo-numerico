import numpy as np


class LU:
    '''
    Classe responsável por executar a decomposição LU de uma matriz.
    
    Uso:
    >>> import numpy as np
    >>> from numerico.linalg import LU
    >>> A = np.array([[ 1, -3,  2],
                      [-2,  8, -1],
                      [ 4, -6,  5]])
    >>> dec = LU()
    >>> L, U, p, sign = dec(A)
    >>> print(L)
    [[ 1.    0.    0.  ]
     [-0.5   1.    0.  ]
     [ 0.25 -0.3   1.  ]]
    >>> print(U)
    [[ 4.  -6.   5. ]
     [ 0.   5.   1.5]
     [ 0.   0.   1.2]]
    >>> print(p)
    [[0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]]
    '''

    def __init__(self, pivoting=True, debug=False, precision=None):
        '''
        Parametros:
        pivoting: bool (padrão True)
            Determina se deve usar pivotação parcial.
        debug: bool (padrão False)
            Determina se deve imprimir linhas das etapas da decomposição LU.
        precision: int (padrão None)
            Determina quantas casas decimais usar no arredondamento a cada 
            iteração externa.
        '''

        self.pivoting = pivoting
        self.debug = debug
        self.precision = precision

    def _setUp(self):
        # Cria as matrizes LU e p.
        self.N = self.matrix.shape[0]
        self.LU = self.matrix.copy()
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
            if pivot == 0:
                raise ZeroDivisionError('Can\'t decompose a singular matrix (det = 0).')
            i += column
        else:
            i = column
            pivot = self.LU[i,i]
            if pivot == 0:
                raise ZeroDivisionError('0 as a pivot found. Please try setting pivoting=True.')
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
        p = self.LU[pivot,pivot]
        mult = self.LU[cur,pivot] / p
        self.LU[cur, pivot] = mult
        self.LU[cur, pivot + 1:] -= mult * self.LU[pivot, pivot + 1:]

    def _round(self):
        # arredonda toda a matriz LU
        if self.precision is not None:
            self.LU = self.LU.round(self.precision)

    def _diagonal_ones(self, matrix):
        # retorna a matriz passada com sua diagonal principal = 1.
        matrix = matrix.copy()
        for i in range(self.N):
            matrix[i, i] = 1
        return matrix

    def _get_lower_trig(self, matrix):
        # retorna o triângulo inferior da matriz dada.
        out = np.array(matrix)
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                out[i, j] = 0
        return out

    @classmethod
    def _get_upper_trig(self, matrix):
        # retorna o triângulo superior da matriz dada.
        out = np.array(matrix)
        for i in range(matrix.shape[0]):
            for j in range(i):
                out[i,j] = 0
        return out

    @property
    def _L(self):
        # retorna a matriz L referente a LU.
        return self._diagonal_ones(self._get_lower_trig(self.LU))

    @property
    def _U(self):
        # retorna a matriz U referente a LU.
        return self._get_upper_trig(self.LU)

    def __call__(self, matrix):
        '''
        Parâmetros:
        matrix: np.array 2d simétrico
            Matriz que se deseja decompor.
        --------
        Retorno:
        L: np.array
            matriz triangular inferior tal que LU = pA
        U: np.array
            matriz triangular superior tal que LU = pA
        p: np.array
            matriz identidade permutada tal que LU = pA. Se não foi usada a 
            pivotação, p é a matriz identidade.
        sign: 1 ou -1
            valor tal que det(A) = sign * det(U), ou seja, sign = (-1) ** t,
            onde t é o número mínimo de trocas na matriz identidate para chegar 
            na matriz p.
        '''

        self.matrix = matrix.astype(float)
        self.swap_count = 0
        self._setUp()
        for pivot_line in range(self.N):
            i, pivot = self._pick_pivot(pivot_line)
            self._swap(i, pivot_line)
            self._round()
            self._show_steps(pivot_line)
            for cur_line in range(pivot_line + 1, self.N):
                self._apply_pivot(pivot_line, cur_line)
        return self._L, self._U, self.p, (-1) ** self.swap_count
