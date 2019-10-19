import numpy as np


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

def bissection(poly, a, b, max_iter=500, toler=1e-4, debug=False, precision=4):
    if poly(a) * poly(b) > 0:
        raise ValueError('f(a)*f(b) must be less than 0.')
    err = abs(b - a)
    if poly(a) > 0: a, b = b, a
    if debug:
        header = ['iter', 'a', 'b', 'x', 'Fa', 'Fb', 'Fx', 'err']
        print(*(str(h).rjust(precision+2) for h in header), sep='\t')
    for i in range(max_iter):
        err /= 2
        x = (a + b) / 2
        if debug:
            values = np.array([a, b, x, poly(a), poly(b), poly(x), err])\
                             .round(precision)
            values = [i] + list(values)
            print(*[str(v).rjust(precision+2) for v in values], sep='\t')
        if err < toler: break
        if poly(x) < 0:
            a = x
        else:
            b = x
    return x, err
