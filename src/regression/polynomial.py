import numpy as np

from src.linalg.cholesky import Cholesky
from src.regression.linear import Linear


class Polynomial(Linear):

    def __init__(self, x, y, rank=1):
        '''
        Parâmetros:
            x: np.array de uma dimensão (input)
            y: np.array de uma dimensão (output)
        '''
        self.rank = rank
        self.n = rank + 1
        self.x, self.y = x.copy(), y.copy()
        self.x_sums = [x.shape[0]]
        self.Y = [y.sum()]
        for i in range(1, 2 * rank + 1):
            x_pow = x**i
            self.x_sums.append((x_pow).sum())
            if i < self.n:
                self.Y.append((x_pow * y).sum())
        self.x_sums, self.Y = np.array(self.x_sums), np.array(self.Y)
        self.setUp()

    def setUp(self):
        self.X = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1):
                self.X[i, j] = self.x_sums[i+j]
                self.X[j, i] = self.x_sums[i+j]

    @property
    def coefs(self):
        if not hasattr(self, '_coefs'):
            chol = Cholesky(self.X)
            out = chol.solve(self.Y)
            out = list(reversed(out))
            self._coefs = np.poly1d(out)
        return self._coefs

    def __call__(self, x_pred):
        return self.coefs(x_pred)
