from math import sqrt

import numpy as np

from .newton_cotes import NewtonCotes
from ..eqroot import newton, briot_ruffini


def legendre_poly():
    x = np.poly1d([1,0])
    p0 = np.poly1d([1])
    p1 = np.poly1d(x)
    yield p0
    yield p1
    i = 1
    while True:
        i += 1
        p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
        p0, p1 = p1, p2
        yield p2

def legendre_roots():
    for n, poly in enumerate(legendre_poly()):
        A = []
        roots = []
        p = poly
        if n % 2 == 1:
            roots.append(0)
            p, zero = briot_ruffini(p, 0)
            A.append(2 / (poly.deriv()(0)**2))
        for i in range(n // 2):
            r, err = newton(p, p.deriv(), 0.1, toler=1e-14)
            roots.append(r)
            roots.append(-r)
            p, zero = briot_ruffini(p, r)
            p, zero = briot_ruffini(p, -r)
            A.append(2 / ((1 - r**2) * poly.deriv()(r)**2))
            A.append(2 / ((1 - r**2) * poly.deriv()(r)**2))
        yield roots, A

coefs = {}
for i, tpl in enumerate(legendre_roots()):
    t, a = tpl
    coefs[i] = {'t': t, 'a': a}
    if i == 10: break


class GaussLegendre(NewtonCotes):
    def __init__(self, start, stop, n, func, debug=False, precision=None):
        self.debug = debug
        self.precision = precision
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

    def _print_row(self, items):
        if self.precision is not None:
            for i, x in enumerate(items):
                try:
                    items[i] = round(x, self.precision)
                except TypeError:
                    pass
        else:
            precision = 16
        print(*(str(i).rjust(self.precision + 3) for i in items), sep='\t')

    def __call__(self):
        a = coefs[self.n]['a']
        t = coefs[self.n]['t']
        out = 0
        i = 0
        if self.debug: self._print_row(['t', 'F(t)', 'A'])
        for i in range(self.n):
            if self.debug: self._print_row([t[i], self._F(t[i]), a[i]])
            out += a[i] * self._F(t[i])
        return out
