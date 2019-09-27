import numpy as np

from ..linalg import solve

class Vandermonde:
    def setUp(self):
        self.n = len(self.x)
        self.rank = self.n - 1
        self.V = np.zeros((self.n, self.n))
        self.V[:,0] = 1
        for i in range(1, self.n):
            self.V[:,i] = self.x ** i

    def validate_points(self):
        if self.x.shape != self.y.shape:
            raise ValueError('x and y must be the same size.')
        if self.x.ndim != 1:
            raise ValueError('Please pass x and y as separate one-dimensional arrays.')

    def __call__(self, x, y):
        self.x, self.y = x, y
        self.validate_points()
        self.setUp()
        coefs = solve(self.V, self.y)
        coefs = list(reversed(coefs))
        coefs = np.array(coefs)
        coefs = np.poly1d(coefs)
        return coefs
