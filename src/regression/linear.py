import numpy as np
from src.typing import Array1D, Array2D

from src.linalg.gauss_decomp import gauss


class Linear:
    def __init__(self, x: Array2D, y: Array1D):
        '''
        Parâmetros:
            x: np.array de duas dimensões. Cada linha é um ponto e cada coluna 
                uma variável.
            y: np.array de uma dimensão (output)
        '''
        self.x, self.y = x.copy(), y.copy()
        self.x_pwr = np.zeros((x.shape[0], x.shape[1] + 1))
        self.x_pwr[:, 0] = 1
        self.x_pwr[:, 1:] = x
        self.n = self.x_pwr.shape[1]
        self.setUp()

    def setUp(self):
        self.X = np.zeros((self.n, self.n))
        self.Y = np.zeros(self.n)
        for i in range(self.n):
            for j in range(i + 1):
                element = self.x_pwr[:, i] @ self.x_pwr[:, j]
                self.X[i, j] = element
                self.X[j, i] = element
            self.Y[i] = self.x_pwr[:, i] @ self.y

    @property
    def coefs(self):
        if not hasattr(self, '_coefs'):
            self._coefs, _det = gauss(self.X, self.Y)
        return self._coefs

    @property
    def D(self):
        if not hasattr(self, '_D'):
            out = 0
            for i in range(len(self.y)):
                out += (self.y[i] - self(self.x[i]))**2
            self._D = out
        return self._D

    @property
    def r(self):
        return self.r2 ** 0.5

    @property
    def r2(self):
        y2_sum = (self.y**2).sum()
        y_sum2 = self.y.sum()**2 / len(self.y)
        return 1 - self.D / (y2_sum - y_sum2)

    def __call__(self, x_pred):
        if np.ndim(x_pred) == 0:
            x_pred = np.array([x_pred])
        return self.coefs[1:] @ x_pred + self.coefs[0]
