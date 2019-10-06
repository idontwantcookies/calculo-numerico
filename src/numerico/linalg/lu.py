import numpy as np


class LU:
    def __init__(self, pivoting=True, debug=False, precision=None):
        self.pivoting = pivoting
        self.debug = debug
        self.precision = precision

    def setup_matrices(self):
        self.N = self.matrix.shape[0]
        self.LU = self.matrix.copy()
        self.p = np.identity(self.N)

    @classmethod
    def swap_rows(cls, matrix, i, j):
        aux = matrix[i].copy()
        matrix[i] = matrix[j]
        matrix[j] = aux

    @classmethod
    def max(cls, row):
        result = 0
        index = 0
        for i, x in enumerate(row):
            if abs(x) > abs(result):
                result = x
                index = i
        return index, result

    def pick_pivot(self, column):
        if self.pivoting:
            pivotal_col = self.LU[column:, column]
            i, pivot = self.max(pivotal_col)
            if pivot == 0:
                raise ZeroDivisionError('Can\'t decompose a singular matrix (det = 0).')
            i += column
        else:
            i = column
            pivot = self.LU[i,i]
            if pivot == 0:
                raise ZeroDivisionError('0 as a pivot found. Please try setting pivoting=True.')
        return i, pivot

    def swap(self, i, j):
        if i != j:
            self.swap_count += 1
            self.swap_rows(self.LU, i, j)
            self.swap_rows(self.p, i, j)

    def show_steps(self, current_pivot):
        if self.debug:
            print(self.LU[current_pivot:])
            print('-' * 80)

    def apply_pivot(self, pivot, cur):
        p = self.LU[pivot,pivot]
        mult = self.LU[cur,pivot] / p
        self.LU[cur, pivot] = mult
        self.LU[cur, pivot + 1:] -= mult * self.LU[pivot, pivot + 1:]

    def round(self):
        if self.precision is not None:
            self.LU = self.LU.round(self.precision)

    def diagonal_ones(self, matrix):
        for i in range(self.N):
            matrix[i, i] = 1
        return matrix

    @classmethod
    def get_lower_trig(cls, matrix):
        out = np.array(matrix)
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                out[i, j] = 0
        return out

    @classmethod
    def get_upper_trig(cls, matrix):
        out = np.array(matrix)
        for i in range(matrix.shape[0]):
            for j in range(i):
                out[i,j] = 0
        return out

    @property
    def L(self):
        return self.diagonal_ones(self.get_lower_trig(self.LU))

    @property
    def U(self):
        return self.get_upper_trig(self.LU)

    def __call__(self, matrix):
        self.matrix = matrix.astype(float)
        self.swap_count = 0
        self.setup_matrices()
        for pivot_line in range(self.N):
            i, pivot = self.pick_pivot(pivot_line)
            self.swap(i, pivot_line)
            self.round()
            self.show_steps(pivot_line)
            for cur_line in range(pivot_line + 1, self.N):
                self.apply_pivot(pivot_line, cur_line)
        return self.L, self.U, self.p, (-1) ** self.swap_count
