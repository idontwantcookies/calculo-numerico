import random


def dim(obj) -> int:
    '''
    Conta o número de dimensões do objeto, assumindo que ele tenha proporções
    constantes, e usa sempre o primeiro elemento do iterável para testar a 
    dimensionalidade.
    '''
    if type(obj) == str: return 0   # avoiding inf recursion and considering str as single element
    try:
        iter(obj)
        obj[0]
        return 1 + dim(obj[0])
    except TypeError:
        return 0
    except IndexError:
        return 0

def all_elements_have_same_length(iterable):
    if dim(iterable) == 0: return True
    for x, y in zip(iterable[1:], iterable[:-1]):
        if dim(x) != dim(y):
            return False
        elif dim(x) > 0 and len(x) != len(y):
            return False
    return all_elements_have_same_length(iterable[0])

def shape(obj, shape_t=None) -> tuple:
    if shape_t is None: shape_t = []
    if dim(obj) == 0: 
        return
    shape_t.append(len(obj))
    shape(obj[0], shape_t)
    return tuple(shape_t)

class Array(list):
    def __init__(self, iterable=None):
        if iterable is None: iterable = []
        iterable = iterable.copy()
        if dim(iterable) > 1:
            for i, x in enumerate(iterable):
                iterable[i] = self.__class__(x)
        super().__init__(iterable)
        self._raise_if_diff_dimensions()

    @classmethod
    def zeros(cls, *shape):
        if len(shape) == 1:
            return cls([0] * shape[0])
        else:
            return cls([cls.zeros(*shape[1:])] * shape[0])

    @classmethod
    def identity(cls, n):
        result = cls.zeros(n, n)
        for i in range(n):
            result[i][i] = 1
        return result

    @classmethod
    def randint(cls, *shape):
        if len(shape) == 1:
            result = cls()
            for i in range(shape[0]):
                result.append(random.randint(0,100))
            return result
        else:
            result = cls()
            for i in range(shape[0]):
                result.append(cls.randint(*shape[1:]))
            return result

    def transpose(self):
        if self.ndim == 1:
            t = self.__class__([self])
            t = t.transpose()
        elif self.ndim == 2:
            t = self.__class__()
            for j in range(self.shape[1]):
                t.append(self.getcol(j))
            return t
        else:
            raise IndexError('Can\'t transpose an array with dim not in (1,2).')
        return t

    def full_equal(self, other):
        if self.ndim != dim(other):
            return False
        elif not all_elements_have_same_length(other):
            return False
        elif len(self) != len(other):
            return False
        elif self.ndim == 1:
            return all(self == other)
        else:
            for i in range(self.shape[0]):
                if not self[i].full_equal(other[i]): return False
            return True

    @property
    def simmetrical(self):
        return self.full_equal(self.transpose())

    def _raise_if_diff_dimensions(self):
        if not all_elements_have_same_length(self):
            raise IndexError('All dimensions must have a constant number of elements.')

    def getcol(self, col):
        result = self.__class__()
        for row in self:
            result.append(row[col])
        return result


    def append(self, *args, **kwargs):
        super().append(*args, **kwargs)
        self._raise_if_diff_dimensions()

    def insert(self, *args, **kwargs):
        super().insert(*args, **kwargs)
        self._raise_if_diff_dimensions()

    @property
    def shape(self):
        return shape(self)

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

    def _raise_if_invalid_matprod(self, other):
        if self.shape[1] != other.shape[0]:
            raise IndexError('Left side must have the same number of columns as rows on the right side for matmul (@).')

    def __matmul__(self, other):
        if dim(other) > 0 and not isinstance(other, self.__class__):
            other = self.__class__(other)
        if dim(self) == dim(other) == 1:    # scalar product
            result = sum(self * other)
        elif dim(self) == dim(other) == 2:
            self._raise_if_invalid_matprod(other)
            result = self.__class__.zeros(self.shape[0], other.shape[1])
            for i in range(len(self)):
                for j in range(other.shape[1]):
                    # print(result)
                    result[i][j] = self[i] @ other.getcol(j)
        elif dim(self) == 2 and dim(other) == 1:
            other = self.__class__([other]).transpose()
            result = (self @ other).getcol(0)
        else:
            self._raise_mult_len_error()
        return result
