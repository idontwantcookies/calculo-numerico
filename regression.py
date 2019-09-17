import pdb
import numpy as np

from vector import Vector
from matrix import Matrix


class LinearRegression:
    def __init__(self, x, y):
        if type(x) == Vector: x = Matrix([x]).transpose()
        if type(y) == Vector: y = Matrix([y]).transpose()
        if x.nrows != y.nrows:
            raise ValueError('x.ncols and y.ncols must be the same size.')
        self.x = x
        self.y = y
        self.sample_size = x.nrows
        self.n_vars = x.ncols
        self.n_regs = y.ncols

    def fit(self):
        self._create_matrices()
        self._setup_matrices()
        self._get_coefficients()
        return self.coefs

    def _create_matrices(self):
        self.A = Matrix.zeros(self.n_vars + 1, self.n_vars + 1)
        self.b = Matrix.zeros(self.n_vars + 1, self.n_regs)

    def _setup_matrices(self):
        self._setup_A_matrix()
        self._setup_b_matrix()

    def _setup_A_matrix(self):
        self.A[0][0] = self.sample_size
        for i in range(self.n_vars):
            s = self.x.getcol(i).sum()
            self.A[0][i + 1] = s
            self.A[i + 1][0] = s
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                self.A[i+1][j+1] = self.x.getcol(i) * self.x.getcol(j)

    def _setup_b_matrix(self):
        for i in range(self.n_regs):
            self.b[0][i] = self.y.getcol(i).sum()
            for j in range(self.n_vars):
                self.b[j+1][i] = self.y.getcol(i) * self.x.getcol(j)

    def _get_coefficients(self):
        self.coefs = self.A.solve(self.b)

    def predict(self, x_values):
        if type(x_values) == Vector: x_values = Matrix([x_values])
        #inserindo uma coluna de 1s para o coeficiente independente
        x_values = x_values.transpose()
        x_values.insert(0, [1] * x_values.ncols)
        x_values = x_values.transpose()
        return x_values * self.coefs

class PolyRegression(LinearRegression):
    def __init__(self, x, y):
        if type(y) == Vector: y = Matrix([y]).transpose()
        if type(x) == Vector and len(x) == y.nrows:
            x = Matrix([x]).transpose()
        elif type(x) == Matrix and x.ncols == 1 and x.nrows == y.nrows:
            pass
        else:
            raise ValueError('x.ncols must be 1 and x.nrows must be the same as y.nrows.')
        self.x = x
        self.y = y
        self.sample_size = x.nrows
        self.n_vars = 1
        self.n_regs = y.ncols

    def _setup_X_matrix(self, degree):
        x = self.x.getcol(0)
        self.x = Matrix([x]).transpose()
        for deg in range(2, 2 * degree):
            self.x.append(x.apply(lambda i: i**deg))
        self.x = self.x.transpose()

    def fit(self, degree=1):
        self._setup_X_matrix(degree)
        return super().fit()

N = 10
x = Matrix.random(N, 1).getcol(0)
y = 2 * x + Vector([3] * N)
r = PolyRegression(x, y)
print(r.fit())
