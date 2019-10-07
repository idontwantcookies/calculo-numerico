import numpy as np

from ..linalg import Cholesky


class Linear:

    def __init__(self, x, y):
        '''
        Parâmetros:
            x: np.array de duas dimensões. Cada linha é um ponto e cada coluna 
                uma variável.
            y: np.array de uma dimensão (output)
        '''
        self.x = np.zeros((x.shape[0], x.shape[1] + 1))
        self.x[:, 0] = 1
        self.x[:, 1:] = x
        self.y = y.copy()
        self.n = self.x.shape[1]
        self.setUp()

    def setUp(self):
        self.X = np.zeros((self.n, self.n))
        self.Y = np.zeros(self.n)
        for i in range(self.n):
            for j in range(i + 1):
                element = self.x[:,i] @ self.x[:,j]
                self.X[i, j] = element
                self.X[j, i] = element
            self.Y[i] = self.x[:,i] @ self.y

    @property
    def coefs(self):
        if not hasattr(self, '_coefs'):
            chol = Cholesky(self.X)
            self._coefs = chol.solve(self.Y)
        return self._coefs

    def __call__(self, x_pred):
        if np.ndim(x_pred) == 0: x_pred = np.array([x_pred])
        return self.coefs[1:] @ x_pred + self.coefs[0]
