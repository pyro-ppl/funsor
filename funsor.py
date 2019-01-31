from __future__ import absolute_import, division, print_function

import functools
import inspect
import itertools
import math
import numbers
import operator
from abc import ABCMeta, abstractmethod

import opt_einsum
import torch
from six import add_metaclass
from six.moves import reduce

DOMAINS = ("real", "positive", "unit_interval")


def is_size(x):
    return isinstance(x, int) or x in DOMAINS


def _log(x):
    return math.log(x) if isinstance(x, numbers.Number) else x.log()


def _exp(x):
    return math.exp(x) if isinstance(x, numbers.Number) else x.exp()


def _logaddexp(x, y):
    raise NotImplementedError('TODO')


_REDUCE_OP_TO_TORCH = {
    operator.add: torch.sum,
    operator.mul: torch.prod,
    operator.and_: torch.all,
    operator.or_: torch.any,
    _logaddexp: torch.logsumexp,
}


def _align_tensors(*args):
    r"""
    Permute multiple tensors before applying a broadcasted op.

    :param \*args: multiple pairs ``(dims, tensor)``, where each ``dims``
        is a tuple of strings naming its tensor's dimensions.
    :return: a pair ``(dims, tensors)`` where all tensors can be
        broadcast together to a single tensor with ``dims``.
    """
    sizes = {}
    for x_dims, x in args:
        assert isinstance(x_dims, tuple)
        assert all(isinstance(d, str) for d in x_dims)
        assert isinstance(x, torch.Tensor)
        assert len(x_dims) == x.dim()
        for dim, size in zip(x_dims, x.shape):
            sizes[dim] = size
    dims = tuple(sorted(sizes))
    tensors = []
    for i, (x_dims, x) in enumerate(args):
        if x_dims != dims:
            x = x.permute(tuple(x_dims.index(d) for d in dims if d in x_dims))
            x = x.reshape(tuple(sizes[d] if d in x_dims else 1 for d in dims))
        assert x.d() == len(dims)
        tensors.append(x)
    return dims, tensors


@add_metaclass(ABCMeta)
class Funsor(object):
    """
    Abstract base class for functional tensors.

    :param tuple dims: A tuple of strings of dimension names.
    :param tuple shape: A tuple of sizes. Each size is either a nonnegative
        integer or a string denoting a continuous domain.
    """
    def __init__(self, dims, shape):
        assert isinstance(dims, tuple)
        assert all(isinstance(d, str) for d in dims)
        assert len(set(dims)) == len(dims)
        assert isinstance(shape, tuple)
        assert all(is_size(s) for s in shape)
        assert len(dims) == len(shape)
        super(Funsor, self).__init__()
        self.dims = dims
        self.shape = shape

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, key):
        return self(*key) if isinstance(key, tuple) else self(key)

    def rename(self, *args, **kwargs):
        """
        Renames dimensions using a dict constructed from
        ``*args, **kwargs``.
        """
        rename = dict(*args, **kwargs)
        dims = tuple(rename.get(d, d) for d in self.dims)
        return self._rename(dims)

    @abstractmethod
    def _rename(self, dims):
        raise NotImplementedError

    @abstractmethod
    def __bool__(self):
        raise NotImplementedError

    @abstractmethod
    def item(self):
        raise NotImplementedError

    def materialize(self, vectorize=True):
        """
        Materializes to a :class:`TorchFunsor`
        """
        for size in self.shape:
            if not isinstance(size, int):
                raise NotImplementedError(
                    "cannot materialize domain {}".format(size))
        if vectorize:
            # fast vectorized computation
            shape = self.shape
            dim = len(shape)
            args = [torch.arange(float(s)).reshape((s,) + (1,) * (dim - i - 1))
                    for i, s in enumerate(shape)]
            tensor = self(*args).item()
        else:
            # slow elementwise computation
            index = itertools.product(*map(range, self.shape))
            tensor = torch.tensor([self(*i).item() for i in index])
            tensor = tensor.reshape(self.shape)
        return TorchFunsor(self.dims, tensor)

    def pointwise_unary(self, op):
        """
        Pointwise unary operation.
        """

        def fn(*args, **kwargs):
            return op(self(*args, **kwargs))

        return LazyFunsor(self.dims, self.shape, fn)

    def pointwise_binary(self, other, op):
        """
        Broadcasted pointwise binary operation.
        """
        if not isinstance(other, Funsor):
            return self.pointwise_unary(lambda a: op(a, other))

        if self.dims == other.dims:
            dims = self.dims
            shape = self.shape

            def fn(*args, **kwargs):
                return op(self(*args, **kwargs), other(*args, **kwargs))

            return LazyFunsor(dims, shape, fn)

        dims = tuple(sorted(set(self.dims + other.dims)))
        sizes = dict(zip(self.dims + other.dims, self.shape + other.shape))
        shape = tuple(sizes[d] for d in dims)

        def fn(*args, **kwargs):
            kwargs.update(zip(dims, args))
            kwargs1 = {d: i for d, i in kwargs.items() if d in self.dims}
            kwargs2 = {d: i for d, i in kwargs.items() if d in other.dims}
            return op(self(**kwargs1), other(**kwargs2))

        return LazyFunsor(dims, shape, fn)

    def reduce(self, op, dim=None):
        """
        Reduce along one or all dimensions.
        """
        if dim is None:
            result = self
            for dim in self.dims:
                result = result.reduce(dim)
            return result

        size = self.shape[self.dims.index(dim)]
        if not isinstance(size, int):
            raise NotImplementedError
        return reduce(op, (self(**{dim: i}) for i in range(size)))

    def __neg__(self):
        return self.pointwise_unary(operator.neg)

    def log(self):
        return self.pointwise_unary(_log)

    def exp(self):
        return self.pointwise_unary(_exp)

    def __add__(self, other):
        return self.pointwise_binary(other, operator.add)

    def __radd__(self, other):
        return self.pointwise_binary(other, operator.add)

    def __sub__(self, other):
        return self.pointwise_binary(other, operator.sub)

    def __rsub__(self, other):
        return self.pointwise_binary(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self.pointwise_binary(other, operator.mul)

    def __rmul__(self, other):
        return self.pointwise_binary(other, operator.mul)

    def __div__(self, other):
        return self.pointwise_binary(other, operator.div)

    def __rdiv__(self, other):
        return self.pointwise_binary(other, lambda a, b: b / a)

    def __pow__(self, other):
        return self.pointwise_binary(other, operator.pow)

    def __rpow__(self, other):
        return self.pointwise_binary(other, lambda a, b: b ** a)

    def __eq__(self, other):
        return self.pointwise_binary(other, operator.eq)

    def sum(self, dim=None):
        return self.reduce(operator.add, dim)

    def prod(self, dim=None):
        return self.reduce(operator.mul, dim)

    def logsumexp(self, dim=None):
        return self.reduce(_logaddexp, dim)

    def all(self, dim=None):
        return self.reduce(operator.and_, dim)

    def any(self, dim=None):
        return self.reduce(operator.or_, dim)


class LazyFunsor(Funsor):
    """
    Funsor backed by a function.

    :param tuple dims: A tuple of strings of dimension names.
    :param tuple shape: A tuple of sizes. Each size is either a nonnegative
        integer or a string denoting a continuous domain.
    :param callable fn: A function defining contents elementwise.
    """
    def __init__(self, dims, shape, fn):
        assert callable(fn)
        super(LazyFunsor, self).__init__(dims, shape)
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            return self
        dims = tuple(d for d in self.dims[len(args):] if d not in kwargs)
        shape = tuple(s for d, s in zip(self.dims, self.shape) if d in dims)
        fn = functools.partial(self.fn, *args, **kwargs)
        return LazyFunsor(dims, shape, fn)

    def _rename(self, dims):
        return LazyFunsor(dims, self.shape, self.__call__)

    def __bool__(self):
        if self.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous")
        return bool(self.fn())

    def item(self):
        if self.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars")
        return self.fn()


class TorchFunsor(Funsor):
    """
    Funsor backed by a PyTorch Tensor.

    :param tuple dims: A tuple of strings of dimension names.
    :param torch.Tensor tensor: A PyTorch tensor of appropriate shape.
    """
    def __init__(self, dims, tensor):
        assert isinstance(tensor, torch.Tensor)
        super(TorchFunsor, self).__init__(dims, tensor.shape)
        self.tensor = tensor

    def __call__(self, *args, **kwargs):
        for pos, arg in enumerate(args):
            if arg is Ellipsis:
                kwargs.update(zip(reversed(self.dims),
                                  reversed(args[1 + pos:])))
                args = args[:pos]
                break
        kwargs.update(zip(self.dims, args))
        result = self
        for key, value in kwargs.items():
            result = result._substitute(key, value)
        return result

    def _substitute(self, dim, value):
        pos = self.dims.index(dim)
        if isinstance(value, numbers.Number):
            dims = self.dims[:pos] + self.dims[1+pos:]
            tensor = self.tensor[(slice(None),) * pos + (value,)]
            return TorchFunsor(dims, tensor)
        if isinstance(value, slice):
            if value == slice(None):
                return self
            start = 0 if value.start is None else value.start
            stop = self.shape[pos] if value.stop is None else value.stop
            step = 1 if value.step is None else value.step
            value = TorchFunsor((dim,), torch.arange(start, stop, step))
            # now fall through to TorchFunsor case.
        if isinstance(value, TorchFunsor):
            dims = self.dims[:pos] + value.dims + self.dims[1+pos:]
            index = [slice(None)] * len(self.dims)
            index[pos] = value.tensor
            for d in value.dims:
                if d != dim and d in self.dims:
                    raise NotImplementedError('TODO')
            tensor = self.tensor[tuple(index)]
            return TorchFunsor(dims, tensor)
        raise NotImplementedError('TODO handle {}'.format(value))

    def _rename(self, dims):
        return TorchFunsor(dims, self.tensor)

    def __bool__(self):
        return bool(self.tensor)

    def item(self):
        return self.tensor.item()

    def materialize(self):
        return self

    def permute(self, dims):
        """
        Create an equivalent :class:`TorchFunsor` whose ``.dims`` are
        the provided dims. Note all dims must be accounted for in the input.

        :param tuple dims: A tuple of strings representing all named dims
            but in a new order.
        :return: A permuted funsor equivalent to self.
        :rtype: TorchFunsor
        """
        assert set(dims) == set(self.dims)
        tensor = self.tensor.permute(tuple(self.dims.index(d) for d in dims))
        return TorchFunsor(dims, tensor)

    def pointwise_unary(self, op):
        return TorchFunsor(self.dims, op(self.tensor))

    def pointwise_binary(self, other, op):
        if not isinstance(other, TorchFunsor):
            return super(TorchFunsor, self).pointwise_binary(other, op)
        if self.dims == other.dims:
            return TorchFunsor(self.dims, op(self.tensor, other.tensor))
        dims, (self_x, other_x) = _align_tensors((self.dims, self.tensor),
                                                 (other.dims, other.tensor))
        return TorchFunsor(dims, op(self_x, other_x))

    def reduce(self, op, dim):
        if op in _REDUCE_OP_TO_TORCH:
            op = _REDUCE_OP_TO_TORCH[op]
            if dim is None:
                return TorchFunsor((), op(self.tensor))
            pos = self.dims.index(dim)
            dims = self.dims[:pos] + self.dims[1 + pos:]
            return TorchFunsor(dims, op(self.tensor, pos))
        return super(TorchFunsor, self).reduce(op, dim)


class PolynomialFunsor(Funsor):
    """
    WIP
    """
    def __init__(self, dims, shape, coefs):
        assert isinstance(coefs, dict)
        super(PolynomialFunsor, self).__init__(dims, shape)
        self.coefs = coefs

    def __getitem__(self, xs):
        xs = dict(zip(self.dims, xs))
        result = 0
        for key, value in self.coefs.items():
            term = value
            for dim in key:
                term = term * xs[dim]
            result = result + term
        return result


class TransformedFunsor(Funsor):
    """
    WIP
    """
    def __init__(self, dims, shape, base_funsor,
                 pre_transforms=None, post_transform=None):
        super(TransformedFunsor, self).__init__(dims, shape)
        self.base_funsor = base_funsor
        if pre_transforms is None:
            pre_transforms = tuple(() for d in dims)
        if post_transform is None:
            post_transform = ()
        self.pre_transforms = pre_transforms
        self.post_transform = post_transform

    def __getitem__(self, key):
        key = list(key)
        for i, transform in enumerate(self.pre_transforms):
            for t in transform:
                key[i] = t(key[i])
        key = tuple(key)
        value = self.base_funsor[key]
        for t in self.post_transform:
            value = t(value)
        return value

    def log(self):
        post_transform = self.post_transforms + (_log,)
        return TransformedFunsor(self.dims, self.shape, self.base_funsor,
                                 self.pre_transforms, post_transform)

    def exp(self):
        post_transform = self.post_transforms + (_exp,)
        return TransformedFunsor(self.dims, self.shape, self.base_funsor,
                                 self.pre_transforms, post_transform)


def contract(dims, *operands, **kwargs):
    r"""
    Sum-product contraction operation.

    :param tuple dims: a tuple of strings of output dimensions.
    :param \*args: multiple :class:`Funsor`s.
    :param str backend: An opt_einsum backend, defaults to 'torch'.
    """
    assert isinstance(dims, tuple)
    assert all(isinstance(d, str) for d in dims)
    assert all(isinstance(x, Funsor) for x in operands)
    kwargs.setdefault('backend', 'torch')
    args = []
    for x in operands:
        x = x.materialize()
        args.extend([x.tensor, x.dims])
    args.append(dims)
    tensor = opt_einsum.contract(*args, **kwargs)
    return TorchFunsor(dims, tensor)


def _lazy(fn, shape):
    args, vargs, kwargs, defaults = inspect.getargspec(fn)
    assert not vargs
    assert not kwargs
    dims = tuple(args)
    return LazyFunsor(dims, shape, fn)


def lazy(*shape):
    """
    Decorator to construct a lazy tensor.
    """
    return functools.partial(_lazy, shape=shape)


__all__ = [
    'DOMAINS',
    'Funsor',
    'LazyFunsor',
    'contract',
    'is_size',
    'lazy',
]
