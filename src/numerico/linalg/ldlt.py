import numpy as np

from .core import Decomposition, successive_substitutions, \
                  retroactive_substitutions

class LDLt(Decomposition):
    def __init__(self, a, precision=None):
        self.a = np.array(a).astype(float)
        self.n = len(a)
        self.precision = precision
        self._execute()

    def _execute(self):
        out = self.a.copy()
        n = len(self.a)
        for j in range(n):
            cum = 0
            for k in range(j):
                cum += out[j, k]**2 * out[k, k]
            out[j,j] -= cum
            r = 1 / out[j,j]
            for i in range(j + 1, n):
                cum = 0
                for k in range(j):
                    cum += out[i,k] * out[k,k] * out[j,k]
                out[i,j] -= cum
                out[i,j] *= r
                out[j,i] = out[i,j]
            if self.precision is not None: out = out.round(self.precision)
        self.ldlt = out

    @property
    def det(self):
        if not hasattr(self, '_det'):
            prod = 1
            for i in range(self.n):
                prod *= self.ldlt[i,i]
            self._det = prod
        return self._det

    def solve(self, b):
        b = np.array(b)
        t = successive_substitutions(self.ldlt, b, diag=False)
        u = t / self.ldlt.diagonal()
        x = retroactive_substitutions(self.ldlt, u, diag=False)
        return x

    def inv(self):
        if not hasattr(self, '_inv'):
            e = np.identity(self.n)
            out = np.zeros_like(self.a).astype(float)
            for i in range(self.n):
                out[:,i] = self.solve(e[:,i])
            self._inv = out
        return self._inv
