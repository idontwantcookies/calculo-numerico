import numpy as np

from .cholesky import Cholesky
from .lu import LU


def square(a):
    '''Testa se um array é matriz quadrada.'''
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def simmetrical(a):
    '''Testa se uma matriz é simétrica fazendo sua transposição.'''
    return (a == a.T).all()

def is_lower_trig(a):
    '''Testa se uma matriz é triangular inferior.
    Complexidade: ~n²/2 comparações (pior caso)'''
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[1]):
            if a[i,j] != 0: return False
    return True

def is_upper_trig(a):
    '''Testa se uma matriz é triangular superior.
    Complexidade: ~n²/2 comparações (pior caso)'''
    for i in range(a.shape[0]):
        for j in range(i):
            if a[i,j] != 0: return False
    return True

def successive_substitutions(a, b, skip_check=False):
    '''
    Faz substituições sucessivas em uma matriz escalonada triangular inferior.
    a: matriz triangular inferior
    b: vetor de coeficientes independentes

    Passe skip_check=True se tiver certeza de que a é triangular inferior, 
    pulando o teste de complexidade ~n²/2.

    Complexidade: ~n²'''

    if not skip_check and not is_lower_trig(a):
        raise ValueError('a must be a lower trig matrix.')
    b = b.copy().astype(a.dtype)
    x = np.zeros_like(b)
    for i in range(a.shape[0]):
        x[i] = (b[i] - (x[:i] * a[i, :i]).sum()) / a[i, i]
    return x

def retroactive_substitutions(a, b, skip_check=False):
    '''
    Faz substituições retroativas em uma matriz escalonada triangular superior.
    a: matriz triangula superiorr
    b: vetor de coeficientes independentes

    Passe skip_check=True se tiver certeza de que a é triangular superior, 
    pulando o teste de complexidade ~n²/2.

    Complexidade: ~n²'''

    if not skip_check and not is_upper_trig(a):
        raise ValueError('a must be an upper trig matrix.')
    b = b.copy().astype(a.dtype)
    x = np.zeros_like(b)
    for i in range(a.shape[0] - 1, -1, -1):
        x[i] = (b[i] - (x[i:] * a[i, i:]).sum()) / a[i, i]
    return x

def det(a):
    '''Retorna o determinante da matriz a, usando a decomposição mais adequada.'''
    try:
        if is_lower_trig(a) or is_upper_trig(a):
            return a.diagonal().prod()
        elif simmetrical(a):
            dec = Cholesky()
            l = dec(a)
            return l.diagonal().prod()**2
        else:
            dec = LU()
            l, u, p, sign = dec(a)
            return sign * u.diagonal().prod()
    except ZeroDivisionError:
        return 0

def decomp(a):
    '''Testa se a é simétrica; se for, decompõe usando cholesky; c.c., decompõe
    usando LU.'''
    if not square(a): raise ValueError('a must be a square matrix.')
    if simmetrical(a):
        dec = Cholesky()
        l = dec(a)
        u = l.T
        p = np.identity(len(a))
        sign = +1
    else:
        dec = LU()
        l, u, p, sign = dec(a)
    return l, u, p, sign

def quicksolve(l, u, b, *args, **kwargs):
    '''Usa substituições retroativas e sucessivas para resolver um sistema do 
    tipo LUx = b, onde L é triangular inferior e U é triangular superior.'''
    if not u.shape == l.shape:
        raise ValueError('L and U must have the same shape.')
    y = successive_substitutions(l, b, *args, **kwargs)
    x = retroactive_substitutions(u, y, *args, **kwargs)
    return x

def solve(a, b):
    '''Decompõe a matriz a em uma triangular superior e outra inferior, e 
    resolve o sistema LUx = pb onde LU = pA.'''
    l, u, p, sign = decomp(a)
    return quicksolve(l, u, p@b, skip_check=True)

def inv(a):
    '''Usa as decomposições LU ou cholesky para achar a inversa da matriz a.'''
    try:
        l, u, e, sign = decomp(a)
    except ZeroDivisionError:
        raise ZeroDivisionError('Can\'t invert a singular matrix (det = 0).')
    n = len(a)
    out = []
    for i in range(n):
        out.append(quicksolve(l, u, e[:, i], skip_check=True))
    return np.array(out).T
