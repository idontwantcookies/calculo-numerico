from math import sqrt, log10, floor
import numpy as np


_DEFAULT_MAX_ITER = 500
_DEFAULT_TOLER = 1e-4

def briot_ruffini(poly, a):
    quot = []
    aux = 0
    for coef in poly:
        aux *= a
        aux += coef
        quot.append(aux)
    remainder = quot[-1]
    quot = quot[:-1]
    return np.poly1d(quot), remainder

def _sign(x):
    return abs(x) // x

def _print_row(items, padding):
    for i, x in enumerate(items):
        try:
            items[i] = round(x, padding)
        except TypeError:
            pass
    print(*(str(i).rjust(padding+3) for i in items), sep='\t')

_bissection_header = ['iter', 'a', 'b', 'x', 'Fa', 'Fb', 'Fx', 'err']
_linear_header = ['iter', 'a', 'b', 'x', 'Fx', 'delta_x']
_quad_header = _linear_header.copy()
_quad_header.insert(3, 'c')

def bissection(func, a, b, max_iter=_DEFAULT_MAX_ITER, toler=_DEFAULT_TOLER, debug=False):
    precision = -floor(log10(toler))
    Fa, Fb = func(a), func(b)
    if Fa * Fb > 0:
        raise ValueError('f(a)*f(b) must be less than 0.')
    err = abs(b - a)
    if Fa > 0: a, b, Fa, Fb = b, a, Fb, Fa
    if debug: _print_row(_bissection_header, precision)
    for i in range(max_iter):
        err /= 2
        x = (a + b) / 2
        Fx = func(x)
        if debug: _print_row([i, a, b, x, Fa, Fb, Fx, err], precision)
        if err < toler: break
        if Fx < 0:
            a, Fa = x, Fx
        else:
            b, Fb = x, Fx
    return x, err

def secant(func, a, b, max_iter=_DEFAULT_MAX_ITER, toler=_DEFAULT_TOLER, debug=False):
    precision = -floor(log10(toler))
    if debug: _print_row(_linear_header, precision)
    Fa, Fb = func(a), func(b)
    if abs(Fa) < abs(Fb): a, b, Fa, Fb = b, a, Fb, Fa
    x, Fx = b, Fb
    for i in range(max_iter):
        delta_x = -Fx / (Fb - Fa) * (b - a)
        x += delta_x; Fx = func(x)
        if debug: _print_row([i, a, b, x, Fx, delta_x], precision)
        if abs(delta_x) < toler and abs(Fx) < toler: break
        a, b, Fa, Fb = b, x, Fb, Fx
    return x, delta_x

def regula_falsi(func, a, b, max_iter=_DEFAULT_MAX_ITER, toler=_DEFAULT_TOLER, debug=False):
    precision = -floor(log10(toler))
    Fa, Fb = func(a), func(b)
    if Fa * Fb > 0:
        raise ValueError('f(a)*f(b) must be less than 0.')
    if Fa > 0: a, b, Fa, Fb = b, a, Fb, Fa
    if debug: _print_row(_linear_header, precision)
    x, Fx = b, Fb
    for i in range(max_iter):
        delta_x = -Fx / (Fb - Fa) * (b - a)
        x += delta_x; Fx = func(x)
        if debug: _print_row([i, a, b, x, Fx, delta_x], precision)
        if abs(delta_x) < toler and abs(func(x)) < toler: break
        if Fx < 0:
            a, Fa = x, Fx
        else:
            b, Fb = x, Fx
    return x, delta_x

def pegasus(func, a, b, max_iter=_DEFAULT_MAX_ITER, toler=_DEFAULT_TOLER, debug=False):
    precision = -floor(log10(toler))
    Fa, Fb = func(a), func(b)
    if debug: _print_row(_linear_header, precision)
    x, Fx = b, Fb
    for i in range(max_iter):
        delta_x = -Fx / (Fb - Fa) * (b - a)
        x += delta_x; Fx = func(x)
        if debug: _print_row([i, a, b, x, Fx, delta_x], precision)
        if abs(delta_x) < toler and abs(func(x)) < toler: break
        if Fx * Fb < 0:
            a, Fa = b, Fb
        else:
            Fa *= Fb / (Fb + Fx)
        b, Fb = x, Fx
    return x, delta_x

def muller(func, a, c, max_iter=_DEFAULT_MAX_ITER, toler=_DEFAULT_TOLER, debug=False):
    precision = -floor(log10(toler))
    Fa, Fc, b = func(a), func(c), (a + c) / 2
    Fb = func(b)
    x, Fx, delta_x = b, Fb, c - a
    if debug: _print_row(_quad_header, precision)
    for i in range(max_iter):
        if abs(Fx) < toler and abs(delta_x) < toler: break
        h1, h2 = c - b, b - a
        r, t = h1 / h2, x
        A = (Fc - (r + 1) * Fb + r * Fa) / (h1 * (h1 + h2))
        B = (Fc - Fb) / h1 - A * h1
        C = Fb
        z = (-B + _sign(B) * sqrt(B**2 - 4 * A * C)) / (2 * A)
        x, delta_x = b + z, x - t
        Fx = func(x)
        if debug: _print_row([i, a, b, c, x, Fx, delta_x], precision)
        if x > b:
            a, Fa = b, Fb
        else:
            c, Fc = b, Fb
        b, Fb = x, Fx
    return x, delta_x

def wijngaarden_dekker_brent(func, a, b, max_iter=_DEFAULT_MAX_ITER, toler=_DEFAULT_TOLER, debug=False):
    precision = -floor(log10(toler))
    Fa, Fb = func(a), func(b)
    if Fa * Fb > 0:
        raise ValueError('Fa*Fb must be less than zero.')
    if debug: _print_row(['iter', 'x', 'Fx', 'z'], precision)
    c, Fc = b, Fb
    for i in range(max_iter):
        if Fb * Fc > 0: 
            c, Fc, d = a, Fa, b - a
            e = d
        elif abs(Fc) < abs(Fb):
            # the source material says it's if instead of elif here, but it does not work with IF.
            a, b, c, Fa, Fb, Fc = b, c, a, Fb, Fc, Fa
        tol = 2 * toler * max(abs(b), 1)
        z = (c - b) / 2
        if debug: _print_row([i, b, Fb, z], precision)
        if abs(z) <= tol or Fb == 0: break
        # pick between interpolation and bissection
        if abs(e) >= tol and abs(Fa) > abs(Fb):
            s = Fb / Fa
            if a == c:
                # linear interp
                p = 2 * z * s
                q = 1 - s
            else:
                # quadratic inverse interp
                q, r = Fa / Fc, Fb / Fc
                p = s * (2 * z * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            else:
                p = -p
            if 2 * p < min(3 * z * q - abs(tol * q), abs(e * q)):
                # accept interpolation
                e, d = d, p / q
            else:
                # uses bissection since interp failed
                d, e = z, z
        else:
            # bissection
            d, e = z, z
        a, Fa = b, Fb
        if abs(d) > tol:
            b += d
        else:
            b += _sign(z) * tol
        Fb = func(b)
    return b, z

def newton(func, dfunc, x0, toler=_DEFAULT_TOLER, max_iter=_DEFAULT_MAX_ITER,
           debug=False):
    return schroder(func, dfunc, x0, 1, toler, max_iter, debug)

def schroder(func, dfunc, x0, m, toler=_DEFAULT_TOLER, max_iter=_DEFAULT_MAX_ITER,
             debug=False):
    precision = -floor(log10(toler))
    Fx, DFx, x = func(x0), dfunc(x0), x0
    if debug:
        _print_row(['iter', 'x', 'Fx', 'DeltaX'], precision)
    for i in range(max_iter):
        delta_x = m * -Fx / DFx
        x += delta_x
        Fx, DFx = func(x), dfunc(x)
        if debug: _print_row([i, x, Fx, delta_x], precision)
        if abs(delta_x) < toler and abs(Fx) < toler or abs(DFx) == 0: break
    return x, delta_x