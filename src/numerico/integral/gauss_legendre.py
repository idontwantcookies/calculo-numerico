from math import sqrt

from .newton_cotes import NewtonCotes


# todo: auto-generate legendre poly A, t using the formulas
# https://en.wikipedia.org/wiki/Legendre_polynomials
# https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature

coefs = {
    1: {'t': [0], 'a':[2]},
    2: {'t': [1/sqrt(3)], 'a': [1]},
    3: {'t': [0, sqrt(3/5)], 'a': [8/9, 5/9]},
    4: {'t': [sqrt(3/7 - 2/7*sqrt(6/5)), sqrt(3/7 + 2/7*sqrt(6/5))],
        'a': [0.5+sqrt(30)/36, 0.5-sqrt(30)/36]},
    5: {'t': [0, 1/3*sqrt(5 - 2*sqrt(10/7)), 1/3*sqrt(5 + 2*sqrt(10/7))],
        'a': [128/225, (322 + 13*sqrt(70))/900, (322 - 13*sqrt(70))/900]}
}


class GaussLegendre(NewtonCotes):
    def __init__(self, start, stop, n, func, debug=False):
        self.debug = debug
        self.func = func
        self.start = start
        self.stop = stop
        self.n = n

    def _x(self, t):
        a = self.start
        b = self.stop
        return ((b - a) / 2) * t + ((b + a) / 2)

    def _F(self, t):
        return (self.stop - self.start) / 2 * self.func(self._x(t))

    def __call__(self):
        a = coefs[self.n]['a']
        t = coefs[self.n]['t']
        out = 0
        i = 0
        if self.n % 2 == 1:
            out = a[0] * self._F(0)
            i += 1
        while i < (self.n + 1) // 2:
            out += a[i] * self._F(t[i])
            out += a[i] * self._F(-t[i])
            i += 1
        return out
