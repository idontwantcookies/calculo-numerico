import numpy as np

from ..eqroot import newton, briot_ruffini


def legendre_poly(n):
    x = np.poly1d([1,0])
    p0 = np.poly1d([1])
    p1 = np.poly1d(x)
    if n == 0:
        return p0
    elif n == 1:
        return p1
    for i in range(2, n + 1):
        p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
        p0, p1 = p1, p2
    return p2

def legendre_roots(n):
    A = []
    roots = []
    poly = legendre_poly(n)
    p = poly
    if n % 2 == 1:
        roots.append(0)
        p, zero = briot_ruffini(p, 0)
        A.append(2 / (poly.deriv()(0)**2))
    for i in range(n // 2):
        r, err = newton(p, p.deriv(), 0.1, toler=1e-10)
        roots.append(r)
        roots.append(-r)
        p, zero = briot_ruffini(p, r)
        p, zero = briot_ruffini(p, -r)
        A.append(2 / ((1 - r**2) * poly.deriv()(r)**2))
    return roots, A
