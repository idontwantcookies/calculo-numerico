import pdb


def dim(obj) -> int:
    '''
    Conta o número de dimensões do objeto, assumindo que ele tenha proporções
    constantes, e usa sempre o primeiro elemento do iterável para testar a 
    dimensionalidade.
    '''
    if type(obj) == str: return 0   # avoiding inf recursion and considering str as single element
    try:
        iter(obj)
        return 1 + dim(obj[0])
    except TypeError:
        return 0

class Array(list):
    @property
    def ndim(self):
        return dim(self)

    def _raise_len_error(self):
        raise IndexError('Both sides must have the same length.')

    def _raise_mult_len_error(self):
        raise IndexError('Matrix multiplication requires compatible 2D and/or 2D arrays.')

    def apply(self, func, iterable=None):
        if iterable is None:
            return self.__class__([func(x) for x in self])
        else:
            return self.__class__([func(x, y) for x, y in zip(self, iterable)])

    def _apply_zero_or_same_dim(self, func, other):
        if dim(other) == 0:
            other = [other] * len(self)
        other = self.__class__(other)
        if len(self) == len(other):
            return self.apply(func, iterable=other)
        else:
            self._raise_len_error()

    def __add__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: x+y, other)

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self.apply(lambda x: -x)

    def __sub__(self, other):
        result = other + -self
        return -result

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: x*y, other)

    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: x/y, other)

    def __itruediv__(self, other):
        return self / other

    def __rtruediv__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: y/x, other)

    def __floordiv__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: x//y, other)

    def __ifloordiv__(self, other):
        return self // other

    def __rfloordiv__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: y//x, other)

    def __mod__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: x%y, other)

    def __imod__(self, other):
        return self % other

    def __rmod__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: y%x, other)

    def __bool__(self):
        return len(self) != 0

    def __invert__(self):
        return self.apply(lambda x: not x if type(x) == bool else ~x)

    def __and__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x & y, other)

    def __rand__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x & y, other)

    def __iand__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x & y, other)

    def __or__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x | y, other)

    def __ror__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x | y, other)

    def __ior__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x | y, other)

    def __xor__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x ^ y, other)

    def __rxor__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x ^ y, other)

    def __ixor__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x ^ y, other)

    def __eq__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x == y, other)

    def __ne__(self, other):
        return self._apply_zero_or_same_dim(lambda x, y: x != y, other)

    def __gt__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: x>y, other)

    def __ge__(self, other):
        return self > other | self == other

    def __lt__(self, other):
        return self._apply_zero_or_same_dim(lambda x,y: x<y, other)

    def __le__(self, other):
        return self < other | self == other

    def __matmul__(self, other):
        if dim(self) == dim(other) == 1:
            return sum(self * other)
        elif dim(self) == dim(other) == 2:
            # TODO matrix multiplication
            pass
        elif dim(self) == 2 and dim(other) == 1:
            # TODO matrix-vector multiplication
            pass
        else:
            self._raise_mult_len_error()
