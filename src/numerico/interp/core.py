from .vandermonde import Vandermonde

def vandermonde(x, y, *args, **kwargs):
    func = Vandermonde(*args, **kwargs)
    return func(x, y)
