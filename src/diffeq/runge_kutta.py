from typing import Callable
from math import log2, cos, sin

import numpy as np
import pandas as pd

Number = float | int
Array = list[Number]
Matrix = list[Array]
Function2to1 = Callable[[Number, Number], Number]
FunctionYAprox = Callable[[Function2to1, Array, Array, Array, Number], Number]

BASE_N = 100
X0 = 0
XN = 10
Y0 = 1

class IVP:
    X: Array
    Y: Array
    F: Array
    f: Function2to1
    h: Number
    N: int

    def __init__(self, f: Function2to1, X: Array, Y: Array, x_n: Number, N: int):
        if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
            raise ValueError("IVP is not well defined.")
        
        if N < len(X):
            raise ValueError("N must be greater than or equal to the length of X.")

        F = []
        for i in range(len(X)):
            F.append(f(X[i], Y[i]))

        self.f = f
        self.X = X
        self.Y = Y
        self.F = F
        self.h = (x_n - X[0]) / N
        self.N = N

    def get_table(self, expected_y: Array | None = None):
        df = pd.DataFrame(zip(self.X, self.Y, self.F), columns="x y dy/dx".split())
        if expected_y:
            df["exact y"] = expected_y
            df["e"] = np.abs(df["exact y"] - df["y"])
        return df

    def global_error(self, expected_Y: Array):
        return abs(self.Y[-1] - expected_Y[-1])


class RungeKutta:
    A: Matrix
    B: Array
    C: Array
    steps: int

    def __init__(self, A: Matrix, B: Array, C: Array):
        if not (len(A) == len(C) == len(B)):
            raise IndexError("Invalid parameters for Runge Kutta")

        for i in range(len(B)):
            if len(A[i]) == i: continue
            raise ValueError(f"A's row {i} must have length {i}, but it has {len(A[i])}: {A[i]}")

        if C[0] != 0:
            raise ValueError(f"RungeKutta does not accept c0 != 0. Received C = {C}")

        self.steps = len(B)
        self.A = A
        self.B = B
        self.C = C

    def get_k(self, ivp: IVP):
        xi, yi = ivp.X[-1], ivp.Y[-1]
        K = []
        for i in range(self.steps):
            delta_x = self.C[i] * ivp.h
            delta_y = ivp.h * sum(Aij * Kj for Aij, Kj in zip(self.A[i], K))
            K.append(ivp.f(xi + delta_x, yi + delta_y))
        return K

    def solve(self, ivp: IVP):
        for i in range(len(ivp.Y), ivp.N + 1):
            K = self.get_k(ivp)
            next_y = ivp.Y[-1] + ivp.h * sum(Bi * Ki for Bi, Ki in zip(self.B, K))
            next_x = ivp.X[0] + i * ivp.h
            ivp.X.append(next_x)
            ivp.Y.append(next_y)
            ivp.F.append(ivp.f(next_x, next_y))


if __name__ == '__main__':
    f = lambda x, y: -y + 2*cos(x)
    pvi_exact_solution = lambda x: cos(x) + sin(x)

    euler = RungeKutta(
        A=[
            []
        ],
        B = [1], C=[0]
    )

    midpoint = RungeKutta(A=[
        [],
        [1/2]
    ], B=[0, 1], C=[0, 1/2])

    heun = RungeKutta(A=[
        [],
        [1]
    ], B=[1/2, 1/2], C=[0, 1])

    X = [X0 + i * (XN - X0) / BASE_N for i in range(BASE_N + 1)]
    exact_solutions_N = [pvi_exact_solution(x) for x in X]

    ivp1 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=BASE_N)
    euler.solve(ivp1)
    df1 = ivp1.get_table()

    ivp2 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=BASE_N)
    midpoint.solve(ivp2)
    df2 = ivp2.get_table()

    ivp3 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=BASE_N)
    heun.solve(ivp3)
    df3 = ivp3.get_table()

    dfN = df1.drop(columns=["dy/dx"]).rename(columns={"y": "y (euler)"})
    dfN["y (midpoint)"] = ivp2.Y
    dfN["y (heun)"] = ivp3.Y
    dfN["y"] = exact_solutions_N

    print(f"N = {BASE_N}")
    print(dfN)
    print()

    X = [X0 + i * (XN - X0)/(2 * BASE_N) for i in range(2*BASE_N + 1)]
    exact_solutions_2N = [pvi_exact_solution(x) for x in X]

    ivp4 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=2*BASE_N)
    euler.solve(ivp4)
    df4 = ivp4.get_table()

    ivp5 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=2*BASE_N)
    midpoint.solve(ivp5)
    df5 = ivp5.get_table()

    ivp6 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=2*BASE_N)
    heun.solve(ivp6)
    df6 = ivp6.get_table()

    df2N = df4.drop(columns=["dy/dx"]).rename(columns={"y": "y (euler)"})
    df2N["y (midpoint)"] = ivp5.Y
    df2N["y (heun)"] = ivp6.Y
    df2N["y"] = exact_solutions_2N

    print(f"N = {2 * BASE_N}")
    print(df2N)
    print()

    rk4 = RungeKutta(A=[
        [],
        [1/2],
        [0, 1/2],
        [0, 0, 1]
    ], B=[1/6, 1/3, 1/3, 1/6], C=[0, 1/2, 1/2, 1])

    ivp7 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=BASE_N)
    rk4.solve(ivp7)
    df7 = ivp7.get_table()

    ivp8 = IVP(f=f, X=[X0], Y=[Y0], x_n=XN, N=2*BASE_N)
    rk4.solve(ivp8)
    df8 = ivp8.get_table()

    df_rk4 = df7.drop(columns=["dy/dx"]).rename(columns={"y": "y (N=100)"})
    df_rk4["y (N=200)"] = df8["y"]

    euler_p = log2(ivp1.global_error(exact_solutions_N) / ivp4.global_error(exact_solutions_2N))
    midpoint_p = log2(ivp2.global_error(exact_solutions_N) / ivp5.global_error(exact_solutions_2N))
    heun_p = log2(ivp3.global_error(exact_solutions_N) / ivp6.global_error(exact_solutions_2N))
    rk4_p = log2(ivp7.global_error(exact_solutions_N) / ivp8.global_error(exact_solutions_2N))

    print(f"Ordem estimada do método de Euler: {euler_p}")
    print(f"Ordem estimada do método do ponto médio (b2 = 1): {midpoint_p}")
    print(f"Ordem estimada do método de Huen (b2 = 1/2): {heun_p}")
    print(f"Ordem estimada do método de RungeKutta de ordem 4: {rk4_p}")
