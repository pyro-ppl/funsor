from __future__ import absolute_import, division, print_function

import functools
import numbers
from collections import OrderedDict

import opt_einsum
import torch

import funsor.ops as ops
from funsor.six import getargspec
from funsor.terms import Align, Funsor, Number, to_funsor
from multipledispatch import dispatch


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
        if x_dims and x_dims != dims:
            x = x.data.permute(tuple(x_dims.index(d) for d in dims if d in x_dims))
            x = x.reshape(tuple(sizes[d] if d in x_dims else 1 for d in dims))
            assert x.dim() == len(dims)
        tensors.append(x)
    return dims, tensors


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

    def __repr__(self):
        return 'Tensor({}, {})'.format(self.dims, self.data)

    def __str__(self):
        if not self.dims:
            return str(self.data.item())
        return 'Tensor({}, {})'.format(self.dims, self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def _eager_subs(self, **kwargs):
        subs = OrderedDict()
        for d in self.dims:
            if d in kwargs:
                subs[d] = kwargs[d]
        if all(isinstance(v, (Number, numbers.Number, slice, Tensor, str)) for v in subs.values()):
            # Substitute one dim at a time.
            conflicts = list(subs.keys())
            result = self
            for i, (dim, value) in enumerate(subs.items()):
                if isinstance(value, Funsor):
                    if not set(value.dims).isdisjoint(conflicts[1 + i:]):
                        raise NotImplementedError(
                            'TODO implement simultaneous substitution')
                result = result._substitute(dim, value)
            return result
        return None  # defer to lazy substitution

    def _substitute(self, dim, value):
        pos = self.dims.index(dim)
        if isinstance(value, Number):
            value = value.data
            # Fall through to numbers.Number case.
        if isinstance(value, numbers.Number):
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
            # Fall through to Tensor case.
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
        raise RuntimeError('{} should be handled by caller'.format(value))

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data.item()

    def align(self, dims, shape=None):
        """
        Create an equivalent :class:`Tensor` whose ``.dims`` are
        the provided dims. Note all dims must be accounted for in the input.

        :param tuple dims: A tuple of strings representing all named dims
            but in a new order.
        :return: A permuted funsor equivalent to self.
        :rtype: Tensor
        """
        if shape is None:
            assert set(dims) == set(self.dims)
            shape = tuple(self.schema[d] for d in dims)
        if dims == self.dims:
            assert shape == self.shape
            return self
        if set(dims) == set(self.dims):
            data = self.data.permute(tuple(self.dims.index(d) for d in dims))
            return Tensor(dims, data)
        # TODO unsqueeze and expand
        return Align(self, dims, shape)

    def _eager_unary(self, op):
        return Tensor(self.dims, op(self.data))

    def reduce(self, op, dims=None):
        if op in ops.REDUCE_OP_TO_TORCH:
            torch_op = ops.REDUCE_OP_TO_TORCH[op]
            self_dims = frozenset(self.dims)
            if dims is None:
                dims = self_dims
            else:
                dims = self_dims.intersection(dims)
            if not dims:
                return self
            if dims == self_dims:
                if op is ops.logaddexp:
                    # work around missing torch.Tensor.logsumexp()
                    return Tensor((), self.data.reshape(-1).logsumexp(0))
                return Tensor((), torch_op(self.data))
            data = self.data
            for pos in reversed(sorted(map(self.dims.index, dims))):
                if op in (ops.min, ops.max):
                    data = getattr(data, op.__name__)(pos)[0]
                else:
                    data = torch_op(data, pos)
            dims = tuple(d for d in self.dims if d not in dims)
            return Tensor(dims, data)
        return super(Tensor, self).reduce(op, dims)

    def contract(self, sum_op, prod_op, other, dims):
        if isinstance(other, Tensor):
            if sum_op is ops.add and prod_op is ops.mul:
                schema = self.schema.copy()
                schema.update(other.schema)
                for d in dims:
                    del schema[d]
                dims = tuple(schema)
                data = opt_einsum.contract(self.data, self.dims, other.data, other.dims, dims,
                                           backend='torch')
                return Tensor(dims, data)

            if sum_op is ops.logaddexp and prod_op is ops.add:
                schema = self.schema.copy()
                schema.update(other.schema)
                for d in dims:
                    del schema[d]
                dims = tuple(schema)
                data = opt_einsum.contract(self.data, self.dims, other.data, other.dims, dims,
                                           backend='pyro.ops.einsum.torch_log')
                return Tensor(dims, data)

        return super(Tensor, self).contract(sum_op, prod_op, other, dims)


@dispatch(object, Tensor, Tensor)
def eager_binary(op, lhs, rhs):
    if lhs.dims == rhs.dims:
        return Tensor(lhs.dims, op(lhs.data, rhs.data))
    dims, (lhs_data, rhs_data) = align_tensors(lhs, rhs)
    return Tensor(dims, op(lhs_data, rhs_data))


@dispatch(object, Tensor, Number)
def eager_binary(op, lhs, rhs):
    return Tensor(lhs.data, op(lhs.data, rhs.data))


@dispatch(object, Number, Tensor)
def eager_binary(op, lhs, rhs):
    return Tensor(rhs.data, op(lhs.data, rhs.data))


@to_funsor.register(torch.Tensor)
def _to_funsor_tensor(x):
    if x.dim():
        raise ValueError("cannot convert non-scalar tensor to funsor")
    return Tensor((), x)


class Arange(Tensor):
    def __init__(self, name, size):
        data = torch.arange(size)
        super(Arange, self).__init__((name,), data)


class Pointwise(Funsor):
    """
    Funsor backed by a PyTorch pointwise function : Tensors -> Tensor.
    """
    def __init__(self, fn, shape=None):
        assert callable(fn)
        dims = tuple(getargspec(fn)[0])
        if shape is None:
            shape = ('real',) * len(dims)
        super(Pointwise, self).__init__(dims, shape)
        self.fn = fn

    def _eager_subs(self, **kwargs):
        args = []
        for name in self.dims[len(args):]:
            if name in kwargs:
                args.append(kwargs[name])
            else:
                break
        if len(args) == len(self.dims) and all(isinstance(x, Tensor) for x in args):
            dims, tensors = align_tensors(*args)
            data = self.fn(*tensors)
            return Tensor(dims, data)
        return None  # defer to lazy substitution


class Function(object):
    """
    Wrapper for PyTorch functions.

    This is mainly created via the :func:`function` decorator.

    :param tuple inputs: a tuple of tuples of input dims (strings).
    :param tuple output: a tuple of output dims (strings).
    :param callable fn: a PyTorch function to wrap.
    """
    def __init__(self, inputs, output, fn):
        assert isinstance(inputs, tuple)
        for input_ in inputs:
            assert isinstance(input_, tuple)
            assert all(isinstance(s, str) for s in input_)
        assert isinstance(output, tuple)
        assert all(isinstance(s, str) for s in output)
        assert callable(fn)
        super(Function, self).__init__()
        self.inputs = inputs
        self.output = output
        self.fn = fn

    def _eager_subs(self, *args):
        assert len(args) == len(self.inputs)

        args = tuple(map(to_funsor, args))
        if all(isinstance(x, (Number, Tensor)) for x in args):
            broadcast_dims = []
            for input_, x in zip(self.inputs, args):
                for d in x.dims:
                    if d not in input_ and d not in broadcast_dims:
                        broadcast_dims.append(d)
            broadcast_dims = tuple(reversed(broadcast_dims))

            args = tuple(x.align(broadcast_dims + input_).data for input_, x in zip(self.inputs, args))
            return Tensor(broadcast_dims + self.output, self.fn(*args))

        return LazyCall(self, args)


class LazyCall(Funsor):
    """
    Value of a :class:`Function` bound to lazy :class:`Funsor`s.

    This is mainly created via the :func:`function` decorator.

    :param Function fn: A wrapped PyTorch function.
    :param tuple args: A tuple of input funsors.
    """
    def __init__(self, fn, args):
        assert isinstance(fn, Function)
        assert isinstance(args, tuple)
        assert all(isinstance(x, Funsor) for x in args)
        schema = OrderedDict()
        for arg in args:
            schema.update(arg.schema)
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(LazyCall, self).__init__(dims, shape)
        self.fn = fn
        self.args = args

    def __call__(self, **subs):
        args = tuple(x(**subs) for x in self.args)
        return self.fn(*args)


def function(*signature):
    """
    Decorator to wrap PyTorch functions.

    :param tuple inputs: a tuple of input dims tuples.
    :param tuple output: a tuple of output dims.

    Example::

        @funsor.function(('a', 'b'), ('b', 'c'), ('a', 'c'))
        def mm(x, y):
            return torch.matmul(x, y)

        @funsor.function(('a',), ('b', 'c'), ('d'))
        def mvn_log_prob(loc, scale_tril, x):
            d = torch.distributions.MultivariateNormal(loc, scale_tril)
            return d.log_prob(x)
    """
    inputs, output = signature[:-1], signature[-1]
    return functools.partial(Function, inputs, output)


__all__ = [
    'Arange',
    'Pointwise',
    'Tensor',
    'align_tensors',
    'function',
]
