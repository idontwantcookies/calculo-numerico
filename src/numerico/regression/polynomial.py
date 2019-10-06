import numpy as np

from ..linalg import solve


class Polynomial:

    def __init__(self, x, y, rank=1):
        '''
        Parâmetros:
            x: np.array de uma dimensão (input)
            y: np.array de uma dimensão (output)
        '''
        self.n = rank + 1
        self.x = [x.shape[0]]
        self.Y = [y.sum()]
        for i in range(1, 2 * rank + 1):
            x_pow = x**i
            self.x.append((x_pow).sum())
            if i < self.n:
                self.Y.append((x_pow * y).sum())
        self.x, self.Y = np.array(self.x), np.array(self.Y)
        self.setUp()

    def setUp(self):
        self.X = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1):
                self.X[i,j] = self.x[i+j]
                self.X[j,i] = self.x[i+j]

    @property
    def coefs(self):
        if not hasattr(self, '_coefs'):
            self._coefs = solve(self.X, self.Y).real
            self._coefs = list(reversed(self._coefs))
            self._coefs = np.poly1d(self._coefs)
        return self._coefs

    def __call__(self, x_pred):
        return self.coefs(x_pred)
