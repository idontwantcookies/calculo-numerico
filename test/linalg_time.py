from time import clock
from statistics import mean

import numpy as np
import pandas as pd

from src.numerico.linalg import LU, Cholesky, LDLt, Jacobi, GaussSeidel, SOR

TOLER = 1e-3

np.seterr(divide='raise', invalid='raise')
pd.set_option('precision', 8)

general = [LU, Jacobi, GaussSeidel]
simmetrical = [Cholesky, LDLt]
special = [SOR]

def get_random_matrices(n=3):
    a = np.random.randint(-n**2, n**2, (n,n))
    for i in range(n):
        a[i][i] = abs(a[i]).sum()
    x = np.random.randint(0,10, n)
    b = a @ x
    return a, b, x

def get_random_simmetrical_matrices(n=3):
    a = np.random.randint(-n**2, n**2, (n,n))
    a = a @ a.T
    for i in range(n):
        a[i][i] = abs(a[i]).sum()
    x = np.random.randint(0,10, n)
    b = a @ x
    return a, b, x

def test_solve_method(dec, a, b, x_exp, *args, **kwargs):
    start = clock()
    x = dec(a, *args, **kwargs).solve(b)
    stop = clock()
    time = stop - start
    err = max(abs((x - x_exp) / x_exp))
    return time, err

def test_non_simmetrical_methods(n=None):
    averages = {m.__name__: {'time':[], 'err':[], 'err_count': 0} for m in general}

    for i in range(30):
        a, b, x = get_random_matrices(n=n)
        for dec in general:
            while True:
                try:
                    time, err = test_solve_method(dec, a, b, x)
                    break
                except FloatingPointError:
                    a, b, x = get_random_matrices(n=n)
            averages[dec.__name__]['time'].append(time)
            averages[dec.__name__]['err'].append(err)
            if err > TOLER: averages[dec.__name__]['err_count'] += 1
    for method in averages.keys():
        averages[method]['time'] = mean(averages[method]['time'])
        averages[method]['err'] = mean(averages[method]['err'])
    return pd.DataFrame(averages).transpose()

def test_simmetrical_methods(n=None):
    averages = {m.__name__: {'time':[], 'err':[], 'err_count': 0} for m in general + simmetrical}
    for i in range(30):
        a, b, x = get_random_simmetrical_matrices(n=n)
        for dec in general + simmetrical:
            while True:
                try:
                    time, err = test_solve_method(dec, a, b, x)
                    break
                except FloatingPointError:
                    a, b, x = get_random_simmetrical_matrices(n=n)
            averages[dec.__name__]['time'].append(time)
            averages[dec.__name__]['err'].append(err)
            if err > TOLER: averages[dec.__name__]['err_count'] += 1
    for method in averages.keys():
        averages[method]['time'] = mean(averages[method]['time'])
        averages[method]['err'] = mean(averages[method]['err'])
    return pd.DataFrame(averages).transpose()

for i in (2, 3, 5, 10, 20, 30):
    print(f'Valores para matrizes quaisquer NxN com N = {i}')
    df = test_non_simmetrical_methods(i)
    df.sort_values(by=['time'], inplace=True)
    print(df)
    print()

for i in (2, 3, 5, 10, 20, 30):
    print(f'Valores para matrizes simétricas NxN com N = {i}')
    df = test_simmetrical_methods(i)
    df.sort_values(by=['time'], inplace=True)
    print(df)
    print()