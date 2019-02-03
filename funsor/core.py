from __future__ import absolute_import, division, print_function

import functools
import inspect
import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from numbers import Number
from weakref import WeakValueDictionary

import opt_einsum
import torch
from six import add_metaclass
from six.moves import reduce

import funsor.ops as ops

DOMAINS = ("real", "positive", "unit_interval")


def align_tensors(*args):
    r"""
    Permute multiple tensors before applying a broadcasted op.

    This is mainly useful for implementing eager funsor operations.

    :param Tensor \*args: Multiple :class:`Tensor`s.
    :return: a pair ``(dims, tensors)`` where tensors are all
        :class:`torch.Tensor`s that can be broadcast together to a single data
        with ``dims``.
    """
    sizes = OrderedDict()
    for x in args:
        sizes.update(x.schema)
    dims = tuple(sizes)
    tensors = []
    for i, x in enumerate(args):
        x_dims, x = x.dims, x.data
        if x_dims != dims:
            x = x.data.permute(tuple(x_dims.index(d) for d in dims if d in x_dims))
            x = x.reshape(tuple(sizes[d] if d in x_dims else 1 for d in dims))
        assert x.dim() == len(dims)
        tensors.append(x)
    return dims, tensors


@add_metaclass(ABCMeta)
class Funsor(object):
    """
    Abstract base class for immutable functional tensors.

    .. note:: Probabilistic methods like :meth:`sample` and :meth:`marginal`
        follow the convention that funsors represent log density functions.
        Thus for example the partition function is given by :meth:`logsumexp`.

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
        self.schema = OrderedDict(zip(dims, shape))

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        raise NotImplementedError

    def __getitem__(self, key):
        # TODO handle slice and Ellipsis here
        return self(*key) if isinstance(key, tuple) else self(key)

    # We avoid __setitem__ due to immutability.

    def __bool__(self):
        if self.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous")
        raise NotImplementedError

    def item(self):
        if self.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars")
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

    def unary(self, op):
        """
        Pointwise unary operation.
        """

        def fn(*args, **kwargs):
            return op(self(*args, **kwargs))

        return Function(self.dims, self.shape, fn)

    def binary(self, op, other):
        """
        Broadcasted pointwise binary operation.
        """
        if not isinstance(other, Funsor):
            return self.unary(lambda a: op(a, other))

        if self.dims == other.dims:
            dims = self.dims
            shape = self.shape

            def fn(*args, **kwargs):
                return op(self(*args, **kwargs), other(*args, **kwargs))

            return Function(dims, shape, fn)

        schema = self.schema.copy()
        schema.update(other.schema)
        dims = tuple(schema)
        shape = tuple(schema.values())

        def fn(*args, **kwargs):
            kwargs.update(zip(dims, args))
            kwargs1 = {d: i for d, i in kwargs.items() if d in self.dims}
            kwargs2 = {d: i for d, i in kwargs.items() if d in other.dims}
            return op(self(**kwargs1), other(**kwargs2))

        return Function(dims, shape, fn)

    def reduce(self, op, dims=None):
        """
        Reduce along all or a subset of dimensions.

        :param callable op: A reduction operation.
        :param set dims: An optional set of dims to reduce.
            If unspecified, all dims will be reduced.
        """
        if dims is None:
            dims = frozenset(self.dims)
        else:
            dims = frozenset(dims).intersection(self.dims)

        result = self
        for dim in dims:
            size = result.schema[dim]
            if not isinstance(size, int):
                raise NotImplementedError('cannot reduce dim {}'.format(dim))
            result = reduce(op, (result(**{dim: i}) for i in range(size)))
        return result

    def argreduce(self, op, dims):
        """
        Reduce along a subset of dimensions,
        keeping track of the values of those dimensions.

        :param callable op: A reduction operation.
        :param tuple dims: A tuple dims to be argreduced.
        :return: A tuple ``(args, remaining)`` where ``args`` is a
            dict mapping a subset of input dims to funsors possibly depending
            on remaining dims, and ``remaining`` is a funsor depending on
            remaing dims.
        :rtype: tuple
        """
        dims = frozenset(dims).intersection(self.dims)
        if not dims:
            return {}, self
        raise NotImplementedError

    def contract(self, sum_op, prod_op, other, dims=None):
        """
        Perform a binary contration, equivalent to binary product followed by a
        sum reduction.

        :param callable sum_op: A reduction operation.
        :param callable prod_op: A binary operation.
        :param Funsor other: Another Funsor.
        :param set dims: An optional set of dims to sum-reduce.
            If unspecified, all dims will be reduced.
        :return: A contracted funsor.
        :rtype: Funsor
        """
        return self.binary(prod_op, other).reduce(sum_op, dims)

    def argcontract(self, sum_op, prod_op, other, dims):
        """
        Perform a binary contration, keeping track of the values of reduced
        dimensions. This is equivalent to binary product followed by a
        sum argreduction.

        :param callable sum_op: A reduction operation.
        :param callable prod_op: A binary operation.
        :param Funsor other: Another Funsor.
        :param set dims: An set of dims to sum-reduce.
        :return: A tuple ``(args, remaining)`` where ``args`` is a
            dict mapping a subset of input dims to funsors possibly depending
            on remaining dims, and ``remaining`` is a funsor depending on
            remaing dims.
        :rtype: tuple
        """
        return self.binary(prod_op, other).argreduce(sum_op, dims)

    # ------------------------------------------------------------------------
    # Subclasses should not override these methods; instead override
    # the generic handlers and fall back to super(...).handler.

    def __invert__(self):
        return self.unary(ops.invert)

    def __neg__(self):
        return self.unary(ops.neg)

    def abs(self):
        return self.unary(ops.abs)

    def sqrt(self):
        return self.unary(ops.sqrt)

    def exp(self):
        return self.unary(ops.exp)

    def log(self):
        return self.unary(ops.log)

    def log1p(self):
        return self.unary(ops.log1p)

    def __add__(self, other):
        return self.binary(ops.add, other)

    def __radd__(self, other):
        return self.binary(ops.add, other)

    def __sub__(self, other):
        return self.binary(ops.sub, other)

    def __rsub__(self, other):
        return self.binary(ops.rsub, other)

    def __mul__(self, other):
        return self.binary(ops.mul, other)

    def __rmul__(self, other):
        return self.binary(ops.mul, other)

    def __truediv__(self, other):
        return self.binary(ops.truediv, other)

    def __rtruediv__(self, other):
        return self.binary(ops.rtruediv, other)

    def __pow__(self, other):
        return self.binary(ops.pow, other)

    def __rpow__(self, other):
        return self.binary(ops.rpow, other)

    def __and__(self, other):
        return self.binary(ops.and_, other)

    def __rand__(self, other):
        return self.binary(ops.and_, other)

    def __or__(self, other):
        return self.binary(ops.or_, other)

    def __ror__(self, other):
        return self.binary(ops.or_, other)

    def __xor__(self, other):
        return self.binary(ops.xor, other)

    def __eq__(self, other):
        return self.binary(ops.eq, other)

    def __ne__(self, other):
        return self.binary(ops.ne, other)

    def __lt__(self, other):
        return self.binary(ops.lt, other)

    def __le__(self, other):
        return self.binary(ops.le, other)

    def __gt__(self, other):
        return self.binary(ops.gt, other)

    def __ge__(self, other):
        return self.binary(ops.ge, other)

    def __min__(self, other):
        return self.binary(ops.min, other)

    def __max__(self, other):
        return self.binary(ops.max, other)

    def sum(self, dims=None):
        return self.reduce(ops.add, dims)

    def prod(self, dims=None):
        return self.reduce(ops.mul, dims)

    def logsumexp(self, dims=None):
        return self.reduce(ops.logaddexp, dims)

    def all(self, dims=None):
        return self.reduce(ops.and_, dims)

    def any(self, dims=None):
        return self.reduce(ops.or_, dims)

    def min(self, dims=None):
        return self.reduce(ops.min, dims)

    def max(self, dims=None):
        return self.reduce(ops.max, dims)

    def argmin(self, dims):
        return self.argreduce(ops.min, dims)

    def argmax(self, dims):
        return self.argreduce(ops.max, dims)

    def marginal(self, dims):
        return self.argreduce(ops.marginal, dims)

    def sample(self, dims):
        """
        Randomly samples from a probability distribution whose
        log probability density is represented by this funsor.

        :param tuple dims: a set of dims to be sampled.
        :return: a tuple ``(values, remaining)`` where ``values`` is a
            dict mapping a subset of input dims to funsors of joint samples
            possibly depending on remaining dims, and ``remaining`` is a funsor
            depending on remaing dims.
        :rtype: tuple
        """
        return self.argreduce(self, ops.sample, dims)


_VARIABLES = WeakValueDictionary()


class Variable(Funsor):
    """
    Funsor representing a single continuous-valued free variable.
    Note discrete variables can be materialized as :class:`Tensor`s.

    .. warning:: Do not construct :class:`Variable`s directly.
        instead use :func:`var`.

    :param str name: A variable name.
    :param size: A size, either an int or a ``DOMAIN``.
    """
    def __init__(self, name, size):
        assert (name, size) not in _VARIABLES, (
            'Do not construct Variables directly; '
            'instead use funsor.variables().')
        super(Variable, self).__init__((name,), (size,))

    @property
    def name(self):
        return self.dims[0]

    def __call__(self, *args, **kwargs):
        kwargs.update(zip(self.dims, args))
        if self.name not in kwargs:
            return self
        value = kwargs[self.name]
        if value is Ellipsis:
            return self
        if isinstance(value, str):
            return var(value, self.shape[0])
        if isinstance(value, Number):
            return
        if isinstance(value, Funsor):
            return value
        raise NotImplementedError('TODO handle {}'.format(value))

    def __bool__(self):
        raise ValueError("bool value of Variable is undefined")

    def item(self):
        raise ValueError("Variable cannot be converted to a Python scalar")

    def materialize(self):
        size = self.shape[0]
        if isinstance(size, int):
            return Tensor(self.dims, torch.arange(size))
        raise ValueError("cannot materialize variable of size {}".format(size))


def var(name, size):
    """
    Constructs a new free variable.

    :param str name: A variable name.
    :param size: A size, either an int or a ``DOMAIN``.
    :return: A new funsor with single dimension ``name``.
    :rtype: Variable (if continuous) or Tensor (if discrete).
    """
    key = (name, size)
    if key in _VARIABLES:
        return _VARIABLES[key]
    if isinstance(size, int):
        result = Tensor((name,), torch.arange(size))
    else:
        result = Variable(name, size)
    _VARIABLES[key] = result
    return result


class Unary(Funsor):
    def __init__(self, op, arg):
        assert callable(op)
        assert isinstance(arg, Funsor)
        super(Unary, self).__init__(arg.dims, arg.shape)
        self.op = op
        self.arg = arg

    def __call__(self, *args, **kwargs):
        return self.arg(*args, **kwargs).unary(self.op)


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
        shape = tuple(self.schema[d] for d in dims)
        fn = functools.partial(self.fn, *args, **kwargs)
        return Function(dims, shape, fn)

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

    def argreduce(self, op, dims):
        dims = frozenset(dims).intersection(self.dims)
        if not dims:
            return {}, self
        return self.materialize().argreduce(op, dims)


def _fun(fn, shape):
    args, vargs, kwargs, defaults = inspect.getargspec(fn)
    assert not vargs
    assert not kwargs
    dims = tuple(args)
    return Function(dims, shape, fn)


def fun(*shape):
    """
    Decorator to construct a :class:`Function`.
    """
    return functools.partial(_fun, shape=shape)


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
        if isinstance(value, str):
            if self.dims[pos] == value:
                return self
            dims = list(self.dims)
            dims[pos] = dim
            return Tensor(tuple(dims), self.data)
        raise NotImplementedError('TODO handle {}'.format(value))

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

    def unary(self, op):
        return Tensor(self.dims, op(self.data))

    def binary(self, op, other):
        if not isinstance(other, Tensor):
            return super(Tensor, self).binary(op, other)
        if self.dims == other.dims:
            return Tensor(self.dims, op(self.data, other.data))
        dims, (self_data, other_data) = align_tensors(self, other)
        return Tensor(dims, op(self_data, other_data))

    def reduce(self, op, dims=None):
        if op in ops.REDUCE_OP_TO_TORCH:
            torch_op = ops.REDUCE_OP_TO_TORCH[op]
            if dims is None:
                # work around missing torch.Tensor.logsumexp()
                if op is ops.logaddexp:
                    return Tensor((), self.data.reshape(-1).logsumexp(0))
                return Tensor((), torch_op(self.data))
            dims = frozenset(dims).intersection(self.dims)
            if not dims:
                return self
            data = self.data
            for pos in reversed(sorted(map(self.dims.index, dims))):
                if op in (ops.min, ops.max):
                    data = getattr(data, op.__name__)(pos)[0]
                else:
                    data = torch_op(data, pos)
            dims = tuple(d for d in self.dims if d not in dims)
            return Tensor(dims, data)
        return super(Tensor, self).reduce(op, dims)

    def argreduce(self, op, dims):
        dims = frozenset(dims).intersection(self.dims)
        if not dims:
            return {}, self

        if op in (ops.min, ops.max):
            if len(dims) == 1:
                dim = next(iter(dims))
                pos = self.dims.index(dim)
                value, remaining = getattr(self.data, op.__name__)
                dims = self.dims[:pos] + self.dims
                return {dim: Tensor(dims, value)}, Tensor(dims, remaining)
            raise NotImplementedError('TODO implement multiway argmin, argmax')

        if op is ops.sample:
            if len(dims) == 1:
                dim, = dims
                pos = self.dims.index(dim)
                shift = self.data.max(pos, keepdim=True)
                probs = (self.data - shift).exp()
                remaining = probs.sum(pos).log() + shift.squeeze(pos)

                probs = probs.transpose(pos, -1)
                value = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
                value = value.reshape(probs.shape[:-1] + (0,))
                value = value.transpose(pos, -1).squeeze(pos)

                dims = self.dims[:pos] + self.dims[1 + pos:]
                return {dim: Tensor(dims, value)}, Tensor(dims, remaining)
            raise NotImplementedError('TODO implement multiway sample')

        raise NotImplementedError('TODO handle {}'.format(op))


def to_funsor(x):
    """
    Convert to a :class:`Funsor`.
    Only :class:`Funsor`s and scalars are accepted.
    """
    if isinstance(x, Funsor):
        return x
    if isinstance(x, torch.Tensor):
        assert x.dim() == 0
        return Tensor((), x)
    if isinstance(x, Number):
        return Tensor((), torch.tensor(x))
    raise ValueError("cannot convert to Funsor: {}".format(x))


def to_torch(x):
    if isinstance(x, Funsor):
        return x.materialize().data
    return x


def contract(*operands, **kwargs):
    r"""
    Sum-product contraction operation.

    :param tuple dims: a tuple of strings of output dimensions. Any input dim
        not requested as an output dim will be summed out.
    :param \*operands: multiple :class:`Funsor`s.
    :param tuple dims: An optional tuple of output dims to preserve.
        Defaults to ``()``, meaning all dims are contracted.
    :param str backend: An opt_einsum backend, defaults to 'torch'.
    :return: A contracted funsor.
    :rtype: Funsor
    """
    assert all(isinstance(x, Funsor) for x in operands)
    dims = kwargs.pop('dims', ())
    assert isinstance(dims, tuple)
    assert all(isinstance(d, str) for d in dims)
    kwargs.setdefault('backend', 'torch')
    args = []
    for x in operands:
        x = x.materialize()
        args.extend([x.data, x.dims])
    args.append(dims)
    data = opt_einsum.contract(*args, **kwargs)
    return Tensor(dims, data)


def argcontract(*operands, **kwargs):
    r"""
    Nary contraction operation, keeping track of the values of reduced dims.

    :param tuple dims: a tuple of strings of output dimensions. Any input dim
        not requested as an output dim will be summed out.
    :param \*operands: multiple :class:`Funsor`s.
    :param tuple dims: An optional tuple of output dims to preserve.
        Defaults to ``()``, meaning all dims are contracted.
    :param str backend: An opt_einsum backend, defaults to 'torch'.
    :return: A tuple ``(args, remaining)`` where ``args`` is a
        dict mapping a subset of input dims to funsors possibly depending
        on remaining dims, and ``remaining`` is a funsor depending on
        remaing dims.
    :rtype: tuple
    """
    raise NotImplementedError('TODO')


def logsumproductexp(*operands):
    return contract(*operands, sum_op=ops.logaddexp, prod_op=ops.add)


__all__ = [
    'DOMAINS',
    'Function',
    'Funsor',
    'Tensor',
    'Variable',
    'contract',
    'fun',
    'logsumproductexp',
    'ops',
    'to_funsor',
    'to_torch',
    'var',
]
