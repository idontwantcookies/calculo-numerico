import numpy as np


class Cholesky:
    def __init__(self, precision=None):
        self.precision = precision

    def setUp(self, a):
        self.a = a
        self.n = self.a.shape[0]
        self.L = np.zeros_like(self.a).astype(complex)

    def set_diagonal_element(self, j):
        part_sum = self.a[j, j] - sum(self.L[j]**2)
        if part_sum == 0:
            raise ZeroDivisionError('Can\'t decompose singular matrix (det = 0).')
        self.L[j, j] = part_sum ** 0.5

    def set_non_diagonal_element(self, i, j):
        pivot = self.L[j, j]
        row = self.L[i, :i]
        col = self.L[j, :i]
        part_sum = row @ col
        self.L[i, j] = (self.a[i, j] - part_sum) / pivot

    def round(self):
        if self.precision is not None:
            self.L = self.L.round(self.precision)

    def __call__(self, a):
        self.setUp(a)
        for j in range(self.n):
            self.set_diagonal_element(j)
            for i in range(j+1, self.n):
                self.set_non_diagonal_element(i, j)
            self.round()
        return self.L
