from __future__ import absolute_import, division, print_function

import functools
import inspect
import itertools
import math
import operator
from abc import ABCMeta, abstractmethod
from numbers import Number

import opt_einsum
import torch
from six import add_metaclass
from six.moves import reduce

DOMAINS = ("real", "positive", "unit_interval")


def _abs(x):
    return abs(x) if isinstance(x, Number) else x.abs()


def _sqrt(x):
    return math.sqrt(x) if isinstance(x, Number) else x.sqrt()


def _exp(x):
    return math.exp(x) if isinstance(x, Number) else x.exp()


def _log(x):
    return math.log(x) if isinstance(x, Number) else x.log()


def _log1p(x):
    return math.log1p(x) if isinstance(x, Number) else x.log1p()


def _pow(x, y):
    result = x ** y
    # work around pytorch shape bug
    if isinstance(x, Number) and isinstance(y, torch.Tensor):
        result = result.reshape(y.shape)
    return result


def _rpow(y, x):
    result = x ** y
    # work around pytorch shape bug
    if isinstance(x, Number) and isinstance(y, torch.Tensor):
        result = result.reshape(y.shape)
    return result


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

    :param \*args: multiple pairs ``(dims, data)``, where each ``dims``
        is a tuple of strings naming its tensor's dimensions.
    :return: a pair ``(dims, tensors)`` where all tensors can be
        broadcast together to a single data with ``dims``.
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
        assert x.dim() == len(dims)
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
        assert all(isinstance(s, int) or s in DOMAINS for s in shape)
        assert len(dims) == len(shape)
        super(Funsor, self).__init__()
        self.dims = dims
        self.shape = shape

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        raise NotImplementedError

    def __getitem__(self, key):
        return self(*key) if isinstance(key, tuple) else self(key)

    @abstractmethod
    def __bool__(self):
        raise NotImplementedError

    @abstractmethod
    def item(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, dims):
        """
        Returns a random sample from a probability distribution whose
        log probability density is represented by this funsor.

        :param tuple dims: a tuple dims to be sampled.
        :return: a tuple of joint samples, with ``len(result) == len(dims)``
            and each result having dims ``set(self.dims) - set(dims)``.
        :rtype: tuple
        """
        raise NotImplementedError

    def materialize(self, vectorize=True):
        """
        Materializes to a :class:`Tensor` if possible;
        raises ``NotImplementedError`` otherwise.
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
            data = self(*args).item()
        else:
            # slow elementwise computation
            index = itertools.product(*map(range, self.shape))
            data = torch.tensor([self(*i).item() for i in index])
            data = data.reshape(self.shape)
        return Tensor(self.dims, data)

    def pointwise_unary(self, op):
        """
        Pointwise unary operation.
        """

        def fn(*args, **kwargs):
            return op(self(*args, **kwargs))

        return Function(self.dims, self.shape, fn)

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

            return Function(dims, shape, fn)

        dims = tuple(sorted(set(self.dims + other.dims)))
        sizes = dict(zip(self.dims + other.dims, self.shape + other.shape))
        shape = tuple(sizes[d] for d in dims)

        def fn(*args, **kwargs):
            kwargs.update(zip(dims, args))
            kwargs1 = {d: i for d, i in kwargs.items() if d in self.dims}
            kwargs2 = {d: i for d, i in kwargs.items() if d in other.dims}
            return op(self(**kwargs1), other(**kwargs2))

        return Function(dims, shape, fn)

    def reduce(self, op, dim=None):
        """
        Reduce along one or all dimensions.
        """
        if dim is None:
            result = self
            for dim in self.dims:
                result = result.reduce(dim)
            return result

        if dim not in self.dims:
            return self
        size = self.shape[self.dims.index(dim)]
        if not isinstance(size, int):
            raise NotImplementedError
        return reduce(op, (self(**{dim: i}) for i in range(size)))

    def __neg__(self):
        return self.pointwise_unary(operator.neg)

    def abs(self):
        return self.pointwise_unary(_abs)

    def sqrt(self):
        return self.pointwise_unary(_sqrt)

    def exp(self):
        return self.pointwise_unary(_exp)

    def log(self):
        return self.pointwise_unary(_log)

    def log1p(self):
        return self.pointwise_unary(_log1p)

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

    def __truediv__(self, other):
        return self.pointwise_binary(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.pointwise_binary(other, lambda a, b: b / a)

    def __pow__(self, other):
        return self.pointwise_binary(other, _pow)

    def __rpow__(self, other):
        return self.pointwise_binary(other, _rpow)

    def __and__(self, other):
        return self.pointwise_binary(other, operator.and_)

    def __rand__(self, other):
        return self.pointwise_binary(other, operator.and_)

    def __or__(self, other):
        return self.pointwise_binary(other, operator.or_)

    def __ror__(self, other):
        return self.pointwise_binary(other, operator.or_)

    def __xor__(self, other):
        return self.pointwise_binary(other, operator.xor)

    def __eq__(self, other):
        return self.pointwise_binary(other, operator.eq)

    def __ne__(self, other):
        return self.pointwise_binary(other, operator.ne)

    def __lt__(self, other):
        return self.pointwise_binary(other, operator.lt)

    def __le__(self, other):
        return self.pointwise_binary(other, operator.le)

    def __gt__(self, other):
        return self.pointwise_binary(other, operator.gt)

    def __ge__(self, other):
        return self.pointwise_binary(other, operator.ge)

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


class Function(Funsor):
    """
    Funsor backed by a Python function.

    :param tuple dims: A tuple of strings of dimension names.
    :param tuple shape: A tuple of sizes. Each size is either a nonnegative
        integer or a string denoting a continuous domain.
    :param callable fn: A function defining contents elementwise.
    """
    def __init__(self, dims, shape, fn):
        assert callable(fn)
        super(Function, self).__init__(dims, shape)
        self.fn = fn

    def __call__(self, *args, **kwargs):
        kwargs = {d: i for d, i in kwargs.items() if d in self.dims}
        if not args and not kwargs:
            return self
        dims = tuple(d for d in self.dims[len(args):] if d not in kwargs)
        shape = tuple(s for d, s in zip(self.dims, self.shape) if d in dims)
        fn = functools.partial(self.fn, *args, **kwargs)
        return Function(dims, shape, fn)

    def sample(self, dims):
        return self.materialize().sample(dims)

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


class Tensor(Funsor):
    """
    Funsor backed by a PyTorch Tensor.

    :param tuple dims: A tuple of strings of dimension names.
    :param torch.Tensor data: A PyTorch tensor of appropriate shape.
    """
    def __init__(self, dims, data):
        assert isinstance(data, torch.Tensor)
        super(Tensor, self).__init__(dims, data.shape)
        self.data = data

    def __call__(self, *args, **kwargs):
        kwargs = {d: i for d, i in kwargs.items() if d in self.dims}

        # Handle Ellipsis notation like x[..., 0].
        for pos, arg in enumerate(args):
            if arg is Ellipsis:
                kwargs.update(zip(reversed(self.dims),
                                  reversed(args[1 + pos:])))
                args = args[:pos]
                break

        # Substitute one dim at a time.
        kwargs.update(zip(self.dims, args))
        result = self
        for key, value in kwargs.items():
            result = result._substitute(key, value)
        return result

    def _substitute(self, dim, value):
        pos = self.dims.index(dim)
        if isinstance(value, Number):
            dims = self.dims[:pos] + self.dims[1+pos:]
            data = self.data[(slice(None),) * pos + (value,)]
            return Tensor(dims, data)
        if isinstance(value, slice):
            if value == slice(None):
                return self
            start = 0 if value.start is None else value.start
            stop = self.shape[pos] if value.stop is None else value.stop
            step = 1 if value.step is None else value.step
            value = Tensor((dim,), torch.arange(start, stop, step))
            # Next fall through to Tensor case.
        if isinstance(value, Tensor):
            dims = self.dims[:pos] + value.dims + self.dims[1+pos:]
            index = [slice(None)] * len(self.dims)
            index[pos] = value.data
            for d in value.dims:
                if d != dim and d in self.dims:
                    raise NotImplementedError('TODO')
            data = self.data[tuple(index)]
            return Tensor(dims, data)
        raise NotImplementedError('TODO handle {}'.format(value))

    def sample(self, dims):
        if len(dims) == 1:
            dim, = dims
            pos = self.dims.index(dim)
            probs = (self.data - self.data.max()).exp()
            probs = probs.transpose(pos, -1)
            data = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
            data = data.reshape(probs.shape[:-1] + (0,))
            data = data.transpose(pos, -1).squeeze(pos)
            dims = self.dims[:pos] + self.dims[1 + pos:]
            return (Tensor(dims, data),)
        raise NotImplementedError('TODO')

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data.item()

    def materialize(self):
        return self

    def permute(self, dims):
        """
        Create an equivalent :class:`Tensor` whose ``.dims`` are
        the provided dims. Note all dims must be accounted for in the input.

        :param tuple dims: A tuple of strings representing all named dims
            but in a new order.
        :return: A permuted funsor equivalent to self.
        :rtype: Tensor
        """
        assert set(dims) == set(self.dims)
        data = self.data.permute(tuple(self.dims.index(d) for d in dims))
        return Tensor(dims, data)

    def pointwise_unary(self, op):
        return Tensor(self.dims, op(self.data))

    def pointwise_binary(self, other, op):
        if not isinstance(other, Tensor):
            return super(Tensor, self).pointwise_binary(other, op)
        if self.dims == other.dims:
            return Tensor(self.dims, op(self.data, other.data))
        dims, (self_x, other_x) = _align_tensors((self.dims, self.data),
                                                 (other.dims, other.data))
        return Tensor(dims, op(self_x, other_x))

    def reduce(self, op, dim):
        if op in _REDUCE_OP_TO_TORCH:
            op = _REDUCE_OP_TO_TORCH[op]
            if dim is None:
                return Tensor((), op(self.data))
            if dim not in self.dims:
                return self
            pos = self.dims.index(dim)
            dims = self.dims[:pos] + self.dims[1 + pos:]
            return Tensor(dims, op(self.data, pos))
        return super(Tensor, self).reduce(op, dim)


class Distribution(Funsor):
    """
    Base class for funsors backed by PyTorch Distributions.

    Ground values are interpreted as log probability densities.

    :param tuple dims: A tuple of strings of dimension names.
    :param tuple shape: A tuple of sizes. Each size is either a nonnegative
        integer or a string denoting a continuous domain.
    :param torch.Distribution dist: a distribution object with
        ``event_dim == 1`` and
        ``event_shape[0] == shape.count('real')``.
    :param torch.Tensor log_normalizer: optional log normalizer
        of shape ``dist.batch_shape``. Defaults to zero.
    """
    def __init__(self, dims, shape, dist, log_normalizer=None):
        assert 'real' in shape
        assert all(isinstance(s, int) or s == 'real' for s in shape)
        assert dist.event_dim == 1
        assert dist.event_shape[0] == shape.count('real')
        if log_normalizer is None:
            log_normalizer = torch.zeros(dist.batch_shape)
        assert log_normalizer.shape == dist.batch_shape
        super(Distribution, self).__init__(dims, shape)
        self.log_normalizer = log_normalizer
        self.dist = dist

    def sample(self, dims):
        if not dims:
            return ()
        int_dims = []
        real_dims = []
        for d, s in zip(self.dims, self.shape):
            (int_dims if isinstance(s, int) else real_dims).append(d)
        if set(dims) & set(int_dims):
            raise NotImplementedError('TODO sample wrt log_normalizer')
        if set(dims) <= set(real_dims):
            result_dims = tuple(d for d in self.dims if d not in dims)
            value = self.dist.rsample()
            if set(dims) == set(real_dims):
                return tuple(
                    Tensor(result_dims, value[..., real_dims.index(d)])
                    for d in dims)
            raise NotImplementedError('TODO condition on partial sample')

    def __bool__(self):
        raise ValueError(
            "bool value of Funsor with more than one value is ambiguous")

    def item(self):
        raise ValueError(
            "only one element Funsors can be converted to Python scalars")

    def logsumexp(self, dim=None):
        if dim is None:
            return self.log_normalizer.logsumexp()
        return super(Distribution, self).logsumexp(dim)


class Normal(Distribution):
    """
    Log density of a batched unnormalized diagonal normal distribution.
    """
    def __init__(self, dims, shape, dist, log_normalizer=None):
        assert isinstance(dist, torch.distributions.Independent)
        assert isinstance(dist.base_dist, torch.distributions.Normal)
        assert isinstance(dist, torch.distributions.Normal)
        super(Normal, self).__init__(dims, shape, dist, log_normalizer)

    def __call__(self, *args, **kwargs):
        kwargs = {d: i for d, i in kwargs.items() if d in self.dims}
        kwargs.update(zip(self.dims, args))
        raise NotImplementedError('TODO')


class MultivariateNormal(Distribution):
    """
    Log density of a batched unnormalized multivariate normal distribution.
    """
    def __init__(self, dims, shape, dist, log_normalizer=None):
        assert isinstance(dist, torch.distributions.MultivariateNormal)
        super(MultivariateNormal, self).__init__(
            dims, shape, dist, log_normalizer)

    def __call__(self, *args, **kwargs):
        kwargs = {d: i for d, i in kwargs.items() if d in self.dims}
        kwargs.update(zip(self.dims, args))
        raise NotImplementedError('TODO')


def contract(dims, *operands, **kwargs):
    r"""
    Sum-product contraction operation.

    :param tuple dims: a tuple of strings of output dimensions. Any input dim
        not requested as an output dim will be summed out.
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
        args.extend([x.data, x.dims])
    args.append(dims)
    data = opt_einsum.contract(*args, **kwargs)
    return Tensor(dims, data)


def _of_shape(fn, shape):
    args, vargs, kwargs, defaults = inspect.getargspec(fn)
    assert not vargs
    assert not kwargs
    dims = tuple(args)
    return Function(dims, shape, fn)


def of_shape(*shape):
    """
    Decorator to construct a :class:`Function`.
    """
    return functools.partial(_of_shape, shape=shape)


__all__ = [
    'DOMAINS',
    'Distribution',
    'Function',
    'Funsor',
    'MultivariateNormal',
    'Normal',
    'Tensor',
    'contract',
    'of_shape',
]
