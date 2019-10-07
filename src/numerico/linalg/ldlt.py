import numpy as np


class LDLt:
    def __init__(self, precision=None):
        self.precision = precision

    def __call__(self, a):
        out = a.copy().astype(float)
        n = len(a)
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
        self.out = out
        return out
