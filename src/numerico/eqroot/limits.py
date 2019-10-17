import numpy as np


class Limits:
    def __init__(self, coef, debug=False):
        self.debug = debug
        coef = coef.copy()
        if coef[0] < 0:
            coef = -coef
        self.coef = coef
        self.n = len(coef)
        self.order = self.n - 1
        self._debug_matrix = []

    def _order(self, coef):
        for i in range(len(coef) - 1, 0, -1):
            if coef[i] != 0: break
        return i

    def _upper_positive_limit(self, coef=None):
        if coef is None: coef = self.coef.copy()
        B = -min(coef)
        k = -1
        order = self._order(coef)
        for i in range(self.n):
            if coef[i] < 0:
                k = i
        if k == -1:
            raise ValueError(self.coef, 'has no real positive roots.')
        out = B / coef[order]
        out **= 1 / (order - k)
        out += 1
        self._debug_matrix.append(np.concatenate([coef, np.array([k, order - k, B, out])]))
        return out

    def _lower_positive_limit(self, coef=None):
        if coef is None: coef = self.coef.copy()
        coef = np.flip(coef)
        return 1 / self._upper_positive_limit(coef)

    def _lower_negative_limit(self, coef=None):
        if coef is None: coef = self.coef.copy()
        for i in range(1, self.n, 2):
            coef[i] = -coef[i]
        try:
            return -self._upper_positive_limit(coef)
        except ValueError as e:
            e.message = self.coef + ' has no real negative roots.'
            raise e

    def _upper_negative_limit(self, coef=None):
        if coef is None: coef = self.coef.copy()
        coef = np.flip(coef)
        return 1 / self._lower_negative_limit(coef)

    @property
    def table(self):
        return np.array(self._debug_matrix).T

    @property
    def pos(self):
        return self._lower_positive_limit(), self._upper_positive_limit()

    @property
    def neg(self):
        return self._lower_negative_limit(), self._upper_negative_limit()
