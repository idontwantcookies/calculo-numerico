import numpy as np

from ..linalg import solve


class Linear:

    def __init__(self, x, y):
        '''
        Parâmetros:
        x: np.array de duas dimensões
        y: np.array de uma dimensão
        '''
        self.x = x.copy()
        self.y = y.copy()
        self.n = self.X.shape[1]
        self.setUp()


    def setUp(self):
        X = np.zeros((self.n + 1, self.n + 1))
        X[0, 0] = self.X
        for i in range(self.n):
            X[0, i+1] = self.X[:,i].sum()
            for j in range(1, self.n):

        self.X = X

    @property
    def coefs(self):
        if not hasattr(self, '_coefs'):
            self._coefs = solve(self.X, self.y)
            self._coefs = list(reversed(self._coefs))
            self._coefs = np.poly1d(self._coefs)
        return self._coefs

    def __call__(self, x_pred):
        return self.coefs(x_pred)
