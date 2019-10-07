import numpy as np

from ..linalg import LU

class Vandermonde:
    
    def __init__(self, x, y):
        self.x, self.y = x.copy(), y.copy()
        self.validate_points()
        self.setUp()

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

    @property
    def coefs(self):
        if not hasattr(self, '_coefs'):
            lu = LU(self.V)
            out = lu.solve(self.y)
            out = list(reversed(out))
            out = np.array(out)
            out = np.poly1d(out)
            self._coefs = out
        return self._coefs

    def __call__(self, x_est):
        return np.polyval(self.coefs, x_est)
