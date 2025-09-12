import pytest

import numpy as np

from src.regression.linear import Linear
from src.regression.polynomial import Polynomial


@pytest.fixture
def x1():
    return np.array([[0.3, 2.7, 4.5, 5.9, 7.8]]).T


@pytest.fixture
def x2():
    return np.array([
        [0, 1, 2, 3, 4],
        [-2, -1, 0, 1, 2]
    ]).T


@pytest.fixture
def x3():
    return np.array([
        [0, 1, 2],
        [-3, 0, 2]
    ]).T


@pytest.fixture
def y():
    return np.array([1.8, 1.9, 3.1, 3.9, 3.3])


@pytest.fixture
def y2():
    return np.array([4.1, 3.1, 3.3])


def test_simple_regression(x1, y):
    reg = Linear(x1, y)
    pred = round(reg(0.5), 4)
    assert pred == 1.7909


def test_multiple_regression(x3, y2):
    reg = Linear(x3, y2)
    pred = round(reg([1, 1]), 4)
    assert pred == 1.9


def test_poly_regression_rank1(x1, y):
    reg = Polynomial(x1[:, 0], y, rank=1)
    pred = round(reg(0.5), 4)
    assert pred == 1.7909


def test_poly_regression_rank2(x1, y):
    reg = Polynomial(x1[:, 0], y, rank=2)
    pred = round(reg(0.5), 4)
    assert pred == 1.6635


def test_poly_regression_interp(x1, y):
    ''' testa se a regressão com 3 pontos produz predições idênticas a y_i,
    ou seja, se a parábola dos mínimos quadrados passa pelos 3 pontos 
    dados.'''
    x = x1[:3, 0]
    y = y[:3]
    reg = Polynomial(x, y, rank=2)
    for i in range(3):
        pred = reg(x[i])
        pred = round(pred, 1)
        assert pred == y[i]


def test_poly_regression_quality(x1, y):
    reg = Polynomial(x1[:, 0], y, rank=1)
    assert round(reg.D, 4) == 0.9289
    assert round(reg.r, 4) == 0.8506
    assert round(reg.r**2, 4) == 0.7235


def test_linear_regression_quality(x1, y):
    reg = Linear(x1, y)
    assert round(reg.D, 4) == 0.9289
    assert round(reg.r, 4) == 0.8506
    assert round(reg.r**2, 4) == 0.7235
