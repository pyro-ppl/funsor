from __future__ import absolute_import, division, print_function

import operator
from numbers import Number

import numpy as np
from multipledispatch import Dispatcher
from six import add_metaclass

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow


class Op(Dispatcher):
    def __init__(self, fn):
        super(Op, self).__init__(fn.__name__)
        # register as default operation
        for nargs in (1, 2):
            default_signature = (object,) * nargs
            self.add(default_signature, fn)

    def __repr__(self):
        return self.__name__

    def __str__(self):
        return self.__name__


class TransformOp(Op):
    def set_inv(self, fn):
        """
        :param callable fn: A function that inputs an arg ``y`` and outputs a
            value ``x`` such that ``y=self(x)``.
        """
        assert callable(fn)
        self.inv = fn
        return fn

    def set_log_abs_det_jacobian(self, fn):
        """
        :param callable fn: A function that inputs two args ``x, y``, where
            ``y=self(x)``, and returns ``log(abs(det(dy/dx)))``.
        """
        assert callable(fn)
        self.log_abs_det_jacobian = fn
        return fn

    @staticmethod
    def inv(x):
        raise NotImplementedError

    @staticmethod
    def log_abs_det_jacobian(x, y):
        raise NotImplementedError


class AssociativeOp(Op):
    pass


class AddOp(AssociativeOp):
    pass


class SubOp(Op):
    pass


class NegOp(Op):
    pass


class DivOp(Op):
    pass


class GetitemMeta(type):
    _cache = {}

    def __call__(cls, offset):
        try:
            return GetitemMeta._cache[offset]
        except KeyError:
            instance = super(GetitemMeta, cls).__call__(offset)
            GetitemMeta._cache[offset] = instance
            return instance


@add_metaclass(GetitemMeta)
class GetitemOp(Op):
    """
    Op encoding an index into one dime, e.g. ``x[:,:,:,y]`` for offset of 3.
    """
    def __init__(self, offset):
        assert isinstance(offset, int)
        assert offset >= 0
        self.offset = offset
        self._prefix = (slice(None),) * offset
        super(GetitemOp, self).__init__(self._default)
        self.__name__ = 'GetitemOp({})'.format(offset)

    def _default(self, x, y):
        return x[self._prefix + (y,)] if self.offset else x[y]


getitem = GetitemOp(0)

eq = Op(operator.eq)
ge = Op(operator.ge)
gt = Op(operator.gt)
invert = Op(operator.invert)
le = Op(operator.le)
lt = Op(operator.lt)
ne = Op(operator.ne)
neg = NegOp(operator.neg)
sub = SubOp(operator.sub)
truediv = DivOp(operator.truediv)

add = AddOp(operator.add)
and_ = AssociativeOp(operator.and_)
mul = AssociativeOp(operator.mul)
or_ = AssociativeOp(operator.or_)
xor = AssociativeOp(operator.xor)


@add.register(object)
def _unary_add(x):
    return x.sum()


@Op
def abs(x):
    return x.abs()


@abs.register(Number)
def _abs(x):
    return _builtin_abs(x)


@Op
def sqrt(x):
    return np.sqrt(x)


@TransformOp
def exp(x):
    return np.exp(x)


@exp.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return add(x)


@TransformOp
def log(x):
    return np.log(x)


@log.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return -add(y)


exp.set_inv(log)
log.set_inv(exp)


@Op
def log1p(x):
    return np.log1p(x)


@Op
def pow(x, y):
    return x ** y


@AssociativeOp
def min(x, y):
    if hasattr(x, '__min__'):
        return x.__min__(y)
    if hasattr(y, '__min__'):
        return y.__min__(x)
    return _builtin_min(x, y)


@AssociativeOp
def max(x, y):
    if hasattr(x, '__max__'):
        return x.__max__(y)
    if hasattr(y, '__max__'):
        return y.__max__(x)
    return _builtin_max(x, y)


@AssociativeOp
def logaddexp(x, y):
    shift = max(x, y)
    return log(exp(x - shift) + exp(y - shift)) + shift


@SubOp
def safesub(x, y):
    if isinstance(y, Number):
        return sub(x, y)


@DivOp
def safediv(x, y):
    if isinstance(y, Number):
        return truediv(x, y)


class ReciprocalOp(Op):
    pass


@ReciprocalOp
def reciprocal(x):
    if isinstance(x, Number):
        return 1. / x
    raise ValueError("No reciprocal for type {}".format(type(x)))


DISTRIBUTIVE_OPS = frozenset([
    (logaddexp, add),
    (add, mul),
    (max, mul),
    (min, mul),
    (max, add),
    (min, add),
])


UNITS = {
    mul: 1.,
    add: 0.,
}


PRODUCT_INVERSES = {
    mul: safediv,
    add: safesub,
}


__all__ = [
    'AddOp',
    'AssociativeOp',
    'DISTRIBUTIVE_OPS',
    'GetitemOp',
    'NegOp',
    'Op',
    'PRODUCT_INVERSES',
    'ReciprocalOp',
    'SubOp',
    'UNITS',
    'abs',
    'add',
    'and_',
    'eq',
    'exp',
    'ge',
    'getitem',
    'gt',
    'invert',
    'le',
    'log',
    'log1p',
    'lt',
    'max',
    'min',
    'mul',
    'ne',
    'neg',
    'or_',
    'pow',
    'safediv',
    'safesub',
    'sqrt',
    'sub',
    'truediv',
    'xor',
]
