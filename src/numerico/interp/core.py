from .vandermonde import Vandermonde
from .lagrange import Lagrange


def vandermonde(x, y, *args, **kwargs):
    func = Vandermonde(*args, **kwargs)
    return func(x, y)
