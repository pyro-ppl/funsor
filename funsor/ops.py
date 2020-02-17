# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import operator
from numbers import Number

import numpy as np
from multipledispatch import Dispatcher

_builtin_abs = abs
_builtin_all = all
_builtin_any = any
_builtin_max = max
_builtin_min = min
_builtin_pow = pow
_builtin_sum = sum


class Op(Dispatcher):
    def __init__(self, fn):
        super(Op, self).__init__(fn.__name__)
        # register as default operation
        for nargs in (1, 2):
            default_signature = (object,) * nargs
            self.add(default_signature, fn)

    def __repr__(self):
        return "ops." + self.__name__

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
        # we cast to np.float64 because np.log(True) returns a np.float16 array
        return np.log(x, dtype=np.float64)
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


logaddexp = LogAddExpOp(_logaddexp)
sample = SampleOp(_logaddexp)


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
    (sample, add),
])


UNITS = {
    mul: 1.,
    add: 0.,
}


PRODUCT_INVERSES = {
    mul: safediv,
    add: safesub,
}


######################
# Numeric Array Ops
######################


all = Op(np.all)
amax = Op(np.amax)
amin = Op(np.amin)
any = Op(np.any)
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


@cat.register(int, [array])
def _cat(dim, *x):
    return np.concatenate(x, axis=dim)


@clamp.register(array, object, object)
def _clamp(x, min, max):
    return np.clip(x, min=min, max=max)


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


# numpy version of scipy.linalg.cho_solve
def _cho_solve(c_and_lower, b):
    c, lower = c_and_lower
    if lower:
        A = c @ np.swapaxes(c, -2, -1)
    else:
        A = np.swapaxes(c, -2, -1) @ c
    return np.linalg.solve(A, b)


@Op
def cholesky_solve(x, y):
    batch_shape = np.broadcast(x[..., 0, 0], y[..., 0, 0]).shape
    xs = np.broadcast_to(x, batch_shape + x.shape[-2:]).reshape((-1,) + x.shape[-2:])
    ys = np.broadcast_to(y, batch_shape + y.shape[-2:]).reshape((-1,) + y.shape[-2:])
    ans = [_cho_solve((y, True), x) for (x, y) in zip(xs, ys)]
    ans = np.stack(ans)
    return ans.reshape(batch_shape + ans.shape[-2:])


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
def is_tensor(x):
    return True if isinstance(x, array) else False


@Op
def logsumexp(x, dim):
    amax = np.amax(x, axis=dim, keepdims=True)
    return np.log(np.sum(np.exp(x - amax), axis=dim)) + amax.squeeze(axis=dim)


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


@Op
def reciprocal(x):
    result = np.clip(np.reciprocal(x), a_max=np.finfo(x.dtype).max)
    return result


@Op
def safediv(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x * np.clip(np.reciprocal(y), a_min=None, a_max=finfo.max)


@Op
def safesub(x, y):
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


class TriangularSolveMeta(type):
    _cache = {}

    def __call__(cls, upper, transpose):
        try:
            return TriangularSolveMeta._cache[(upper, transpose)]
        except KeyError:
            instance = super(TriangularSolveMeta, cls).__call__(upper, transpose)
            TriangularSolveMeta._cache[(upper, transpose)] = instance
            return instance


# numpy version of scipy.linalg.solve_triangular
def _solve_triangular(a, b, trans=0, lower=False):
    if trans:
        a = np.swapaxes(a, -2, -1)
    return np.linalg.solve(a, b)


class TriangularSolveOp(Op, metaclass=TriangularSolveMeta):
    def __init__(self, upper, transpose):
        assert isinstance(upper, bool)
        assert isinstance(transpose, bool)
        self.upper = upper
        self.transpose = transpose
        super(TriangularSolveOp, self).__init__(self._default)
        self.__name__ = f'TriangularSolveOp(upper={upper},transpose={transpose})'

    def _default(self, x, y):
        batch_shape = np.broadcast(x[..., 0, 0], y[..., 0, 0]).shape
        xs = np.broadcast_to(x, batch_shape + x.shape[-2:]).reshape((-1,) + x.shape[-2:])
        ys = np.broadcast_to(y, batch_shape + y.shape[-2:]).reshape((-1,) + y.shape[-2:])
        ans = [_solve_triangular(y, x, trans=int(self.transpose), lower=not self.upper)
               for (x, y) in zip(xs, ys)]
        ans = np.stack(ans)
        return ans.reshape(batch_shape + ans.shape[-2:])


def triangular_solve(x, y, upper=False, transpose=False):
    return TriangularSolveOp(upper, transpose)(x, y)


@Op
def unsqueeze(x, dim):
    return np.expand_dims(x, axis=dim)


__all__ = [
    'AddOp',
    'AssociativeOp',
    'DISTRIBUTIVE_OPS',
    'ExpOp',
    'GetitemOp',
    'LogAddExpOp',
    'LogOp',
    'NegOp',
    'Op',
    'PRODUCT_INVERSES',
    'ReciprocalOp',
    'SampleOp',
    'SubOp',
    'ReshapeOp',
    'UNITS',
    'abs',
    'add',
    'all',
    'amax',
    'amin',
    'and_',
    'any',
    'cat',
    'cholesky',
    'cholesky_inverse',
    'cholesky_solve',
    'clamp',
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
    'or_',
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
""".format("\n".join(f".. autodata:: {_name}\n"
                     for _name in __all__ if isinstance(globals()[_name], Op)))
