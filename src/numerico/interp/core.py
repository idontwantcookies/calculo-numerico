from .vandermonde import Vandermonde
from .lagrange import Lagrange
from .newton import Newton
from .point_choice import choose_points


def vandermonde(x, y, *args, **kwargs):
    func = Vandermonde(*args, **kwargs)
    return func(x, y)
