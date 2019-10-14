from math import sqrt

from .newton_cotes import NewtonCotes


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
    pass
