# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import operator
from numbers import Number

import numpy as np
from multipledispatch import Dispatcher

from .op import DISTRIBUTIVE_OPS, PRODUCT_INVERSES, UNITS, Op, TransformOp

_builtin_abs = abs
_builtin_all = all
_builtin_any = any
_builtin_max = max
_builtin_min = min
_builtin_pow = pow
_builtin_sum = sum


# FIXME Most code assumes this is an AssociativeCommutativeOp.
class AssociativeOp(Op):
    pass


class AddOp(AssociativeOp):
    pass


class MulOp(AssociativeOp):
    pass


class MatmulOp(Op):  # Associtive but not commutative.
    pass


class LogAddExpOp(AssociativeOp):
    pass


class SampleOp(LogAddExpOp):
    pass


class SubOp(Op):
    pass


class NegOp(Op):
    pass


class DivOp(Op):
    pass


class NullOp(AssociativeOp):
    """Placeholder associative op that unifies with any other op"""
    pass


@NullOp
def nullop(x, y):
    raise ValueError("should never actually evaluate this!")


class ReshapeMeta(type):
    _cache = {}

    def __call__(cls, shape):
        shape = tuple(shape)
        try:
            return ReshapeMeta._cache[shape]
        except KeyError:
            instance = super().__call__(shape)
            ReshapeMeta._cache[shape] = instance
            return instance


class ReshapeOp(Op, metaclass=ReshapeMeta):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(self._default)

    def __reduce__(self):
        return ReshapeOp, (self.shape,)

    def _default(self, x):
        return x.reshape(self.shape)


class GetitemMeta(type):
    _cache = {}

    def __call__(cls, offset):
        try:
            return GetitemMeta._cache[offset]
        except KeyError:
            instance = super(GetitemMeta, cls).__call__(offset)
            GetitemMeta._cache[offset] = instance
            return instance


class GetitemOp(Op, metaclass=GetitemMeta):
    """
    Op encoding an index into one dimension, e.g. ``x[:,:,y]`` for offset of 2.
    """
    def __init__(self, offset):
        assert isinstance(offset, int)
        assert offset >= 0
        self.offset = offset
        self._prefix = (slice(None),) * offset
        super(GetitemOp, self).__init__(self._default)
        self.__name__ = 'GetitemOp({})'.format(offset)

    def __reduce__(self):
        return GetitemOp, (self.offset,)

    def _default(self, x, y):
        return x[self._prefix + (y,)] if self.offset else x[y]


getitem = GetitemOp(0)
abs = Op(_builtin_abs)
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
mul = MulOp(operator.mul)
matmul = MatmulOp(operator.matmul)
or_ = AssociativeOp(operator.or_)
xor = AssociativeOp(operator.xor)


@add.register(object)
def _unary_add(x):
    return x.sum()


@Op
def sqrt(x):
    return np.sqrt(x)


class ExpOp(TransformOp):
    pass


@ExpOp
def exp(x):
    return np.exp(x)


@exp.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return add(x)


class LogOp(TransformOp):
    pass


@LogOp
def log(x):
    if isinstance(x, bool) or (isinstance(x, np.ndarray) and x.dtype == 'bool'):
        return np.where(x, 0., float('-inf'))
    with np.errstate(divide='ignore'):  # skip the warning of log(0.)
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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def _logaddexp(x, y):
    if hasattr(x, "__logaddexp__"):
        return x.__logaddexp__(y)
    if hasattr(y, "__rlogaddexp__"):
        return y.__logaddexp__(x)
    shift = max(x, y)
    return log(exp(x - shift) + exp(y - shift)) + shift


logaddexp = LogAddExpOp(_logaddexp, name="logaddexp")
sample = SampleOp(_logaddexp, name="sample")


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


DISTRIBUTIVE_OPS.add((logaddexp, add))
DISTRIBUTIVE_OPS.add((add, mul))
DISTRIBUTIVE_OPS.add((max, mul))
DISTRIBUTIVE_OPS.add((min, mul))
DISTRIBUTIVE_OPS.add((max, add))
DISTRIBUTIVE_OPS.add((min, add))
DISTRIBUTIVE_OPS.add((sample, add))

UNITS[mul] = 1.
UNITS[add] = 0.

PRODUCT_INVERSES[mul] = safediv
PRODUCT_INVERSES[add] = safesub


######################
# Numeric Array Ops
######################


all = Op(np.all)
amax = Op(np.amax)
amin = Op(np.amin)
any = Op(np.any)
astype = Dispatcher("ops.astype")
cat = Dispatcher("ops.cat")
clamp = Dispatcher("ops.clamp")
diagonal = Dispatcher("ops.diagonal")
einsum = Dispatcher("ops.einsum")
full_like = Op(np.full_like)
prod = Op(np.prod)
stack = Dispatcher("ops.stack")
sum = Op(np.sum)
transpose = Dispatcher("ops.transpose")

array = (np.ndarray, np.generic)


@astype.register(array, str)
def _astype(x, dtype):
    return x.astype(dtype)


@cat.register(int, [array])
def _cat(dim, *x):
    return np.concatenate(x, axis=dim)


@clamp.register(array, object, object)
def _clamp(x, min, max):
    return np.clip(x, a_min=min, a_max=max)


@Op
def cholesky(x):
    """
    Like :func:`numpy.linalg.cholesky` but uses sqrt for scalar matrices.
    """
    if x.shape[-1] == 1:
        return np.sqrt(x)
    return np.linalg.cholesky(x)


@Op
def cholesky_inverse(x):
    """
    Like :func:`torch.cholesky_inverse` but supports batching and gradients.
    """
    return cholesky_solve(new_eye(x, x.shape[:-1]), x)


@Op
def cholesky_solve(x, y):
    y_inv = np.linalg.inv(y)
    A = np.swapaxes(y_inv, -2, -1) @ y_inv
    return A @ x


@Op
def detach(x):
    return x


@diagonal.register(array, int, int)
def _diagonal(x, dim1, dim2):
    return np.diagonal(x, axis1=dim1, axis2=dim2)


@einsum.register(str, [array])
def _einsum(x, *operand):
    return np.einsum(x, *operand)


@Op
def expand(x, shape):
    prepend_dim = len(shape) - np.ndim(x)
    assert prepend_dim >= 0
    shape = shape[:prepend_dim] + tuple(dx if size == -1 else size
                                        for dx, size in zip(np.shape(x), shape[prepend_dim:]))
    return np.broadcast_to(x, shape)
    return np.broadcast_to(x, shape)


@Op
def finfo(x):
    return np.finfo(x.dtype)


@Op
def is_numeric_array(x):
    return True if isinstance(x, array) else False


@Op
def logsumexp(x, dim):
    amax = np.amax(x, axis=dim, keepdims=True)
    # treat the case x = -inf
    amax = np.where(np.isfinite(amax), amax, 0.)
    return log(np.sum(np.exp(x - amax), axis=dim)) + amax.squeeze(axis=dim)


@max.register(array, array)
def _max(x, y):
    return np.maximum(x, y)


@max.register((int, float), array)
def _max(x, y):
    return np.clip(y, a_min=x, a_max=None)


@max.register(array, (int, float))
def _max(x, y):
    return np.clip(x, a_min=y, a_max=None)


@min.register(array, array)
def _min(x, y):
    return np.minimum(x, y)


@min.register((int, float), array)
def _min(x, y):
    return np.clip(y, a_min=None, a_max=x)


@min.register(array, (int, float))
def _min(x, y):
    return np.clip(x, a_min=None, a_max=y)


@Op
def new_arange(x, stop):
    return np.arange(stop)


@new_arange.register(array, int, int, int)
def _new_arange(x, start, stop, step):
    return np.arange(start, stop, step)


@Op
def new_zeros(x, shape):
    return np.zeros(shape, dtype=x.dtype)


@Op
def new_eye(x, shape):
    n = shape[-1]
    return np.broadcast_to(np.eye(n), shape + (n,))


@Op
def permute(x, dims):
    return np.transpose(x, axes=dims)


@reciprocal.register(array)
def _reciprocal(x):
    result = np.clip(np.reciprocal(x), a_max=np.finfo(x.dtype).max)
    return result


@safediv.register(object, array)
def _safediv(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x * np.clip(np.reciprocal(y), a_min=None, a_max=finfo.max)


@safesub.register(object, array)
def _safesub(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x + np.clip(-y, a_min=None, a_max=finfo.max)


@stack.register(int, [array])
def _stack(dim, *x):
    return np.stack(x, axis=dim)


@transpose.register(array, int, int)
def _transpose(x, dim1, dim2):
    return np.swapaxes(x, dim1, dim2)


@Op
def triangular_solve(x, y, upper=False, transpose=False):
    if transpose:
        y = np.swapaxes(y, -2, -1)
    return np.linalg.inv(y) @ x


@Op
def unsqueeze(x, dim):
    return np.expand_dims(x, axis=dim)


__all__ = [
    'AddOp',
    'AssociativeOp',
    'DivOp',
    'ExpOp',
    'GetitemOp',
    'LogAddExpOp',
    'LogOp',
    'MatmulOp',
    'MulOp',
    'NegOp',
    'NullOp',
    'ReciprocalOp',
    'SampleOp',
    'SubOp',
    'ReshapeOp',
    'abs',
    'add',
    'all',
    'amax',
    'amin',
    'and_',
    'any',
    'astype',
    'cat',
    'cholesky',
    'cholesky_inverse',
    'cholesky_solve',
    'clamp',
    'detach',
    'diagonal',
    'einsum',
    'eq',
    'exp',
    'expand',
    'finfo',
    'full_like',
    'ge',
    'getitem',
    'gt',
    'invert',
    'is_numeric_array',
    'le',
    'log',
    'log1p',
    'logaddexp',
    'logsumexp',
    'lt',
    'matmul',
    'max',
    'min',
    'mul',
    'ne',
    'neg',
    'new_arange',
    'new_eye',
    'new_zeros',
    'nullop',
    'or_',
    'permute',
    'pow',
    'prod',
    'reciprocal',
    'safediv',
    'safesub',
    'sample',
    'sigmoid',
    'sqrt',
    'stack',
    'sub',
    'sum',
    'transpose',
    'triangular_solve',
    'truediv',
    'unsqueeze',
    'xor',
]

__doc__ = """
Built-in operations
-------------------

{}

Operation classes
-----------------
""".format("\n".join(".. autodata:: {}\n".format(_name)
                     for _name in __all__ if isinstance(globals()[_name], Op)))
