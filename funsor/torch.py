from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import torch
from six import add_metaclass

import funsor.ops as ops
from funsor.domains import Domain, ints
from funsor.interpreter import eager, interpret
from funsor.six import getargspec
from funsor.terms import Binary, Funsor, FunsorMeta, Number, to_funsor


def align_tensors(*args):
    r"""
    Permute multiple tensors before applying a broadcasted op.

    This is mainly useful for implementing eager funsor operations.

    :param funsor.terms.Funsor \*args: Multiple :class:`Tensor`s and
        :class:`~funsor.terms.Number`s.
    :return: a pair ``(inputs, tensors)`` where tensors are all
        :class:`torch.Tensor`s that can be broadcast together to a single data
        with given ``inputs``.
    :rtype: tuple
    """
    # TODO update this
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


class TensorMeta(FunsorMeta):
    """
    Wrapper to fill in default args and convert between OrderedDict and tuple.
    """
    def __call__(cls, data, inputs=None, dtype="real"):
        if inputs is None:
            inputs = tuple()
        elif isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        return super(TensorMeta, cls).__call__(data, inputs, dtype)


@to_funsor.register(torch.Tensor)
@add_metaclass(TensorMeta)
class Tensor(Funsor):
    """
    Funsor backed by a PyTorch Tensor.

    :param tuple dims: A tuple of strings of dimension names.
    :param torch.Tensor data: A PyTorch tensor of appropriate shape.
    """
    def __init__(self, data, inputs=None, dtype="real"):
        assert isinstance(data, torch.Tensor)
        assert isinstance(inputs, tuple)
        inputs = OrderedDict(inputs)
        input_dim = sum(i.num_elements() for i in inputs.values())
        output = Domain(data.shape[input_dim:], dtype)
        super(Tensor, self).__init__(inputs, output)
        self.data = data

    def __repr__(self):
        if self.output != "real":
            return 'Tensor({}, {}, {})'.format(self.data, self.inputs, repr(self.dtype))
        elif self.inputs:
            return 'Tensor({}, {})'.format(self.data, self.inputs)
        else:
            return 'Tensor({})'.format(self.data)

    def __str__(self):
        if self.dtype != "real":
            return 'Tensor({}, {}, {})'.format(self.data, self.inputs, repr(self.dtype))
        elif self.inputs:
            return 'Tensor({}, {})'.format(self.data, self.inputs)
        else:
            return str(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data.item()

    def eager_subs(self, subs):
        # TODO handle variables.
        if all(isinstance(v, (Number, Tensor)) and not v.inputs for v in subs.values()):
            # Substitute one variable at a time.
            conflicts = list(subs.keys())
            result = self
            for i, (key, value) in enumerate(subs.items()):
                if isinstance(value, Funsor):
                    if not set(value.inputs).isdisjoint(conflicts[1 + i:]):
                        raise NotImplementedError(
                            'TODO implement simultaneous substitution')
                result = result._substitute(key, value)
            return result

        return None  # defer to default implementation

    def _substitute(self, name, value):
        pos = self.dims.index()
        if isinstance(value, Number):
            value = value.data
            dims = self.dims[:pos] + self.dims[1+pos:]
            data = self.data[(slice(None),) * pos + (value,)]
            return interpret(Tensor, dims, data)
        if isinstance(value, Tensor):
            dims = self.dims[:pos] + value.dims + self.dims[1+pos:]
            index = [slice(None)] * len(self.dims)
            index[pos] = value.data
            for key in value.inputs:
                if key != name and key in self.inputs:
                    raise NotImplementedError('TODO')
            data = self.data[tuple(index)]
            return interpret(Tensor, dims, data)
        raise RuntimeError('{} should be handled by caller'.format(value))

    def eager_unary(self, op):
        return interpret(Tensor, op(self.data), self.inputs, self.dtype)

    def eager_reduce(self, op, dims=None):
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
                    data = self.data.reshape(-1).logsumexp(0)
                    return interpret(Tensor, data)
                return interpret(Tensor, torch_op(self.data))
            data = self.data
            for pos in reversed(sorted(map(self.dims.index, dims))):
                if op in (ops.min, ops.max):
                    data = getattr(data, op.__name__)(pos)[0]
                else:
                    data = torch_op(data, pos)
            dims = tuple(d for d in self.dims if d not in dims)
            return interpret(Tensor, data, dims)
        return super(Tensor, self).reduce(op, dims)


@eager.register(Binary, object, Tensor, Number)
def eager_binary_tensor_number(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    return interpret(Tensor, data, lhs.inputs, lhs.dtype)


@eager.register(Binary, object, Number, Tensor)
def eager_binary_number_tensor(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    return interpret(Tensor, data, rhs.inputs, rhs.dtype)


@eager.register(Binary, object, Tensor, Tensor)
def eager_binary_tensor_tensor(op, lhs, rhs):
    assert lhs.dtype == rhs.dtype
    if lhs.inputs == rhs.inputs:
        inputs = lhs.inputs
        data = op(lhs.data, rhs.data)
    else:
        inputs, (lhs_data, rhs_data) = align_tensors(lhs, rhs)
        data = op(lhs_data, rhs_data)
    return interpret(Tensor, data, inputs, lhs.dtype)


class Arange(Tensor):
    """
    Helper to create a named :func:`torch.arange` funsor.

    :param str name: A variable name.
    :param int size: A size.
    """
    def __init__(self, name, size):
        data = torch.arange(size)
        inputs = OrderedDict([(name, ints(size))])
        super(Arange, self).__init__(data, inputs)


class Function(object):
    """
    Funsor wrapped by a for PyTorch function.

    Functions are assumed to support broadcasting.

    These are often created via the :func:`function` decorator.

    :param tuple inputs: A tuple of domains of function arguments.
    :param funsor.domains.Domain output: An output domain.
    :param callable fn: A PyTorch function to wrap.
    """
    def __init__(self, inputs, output, fn):
        assert isinstance(inputs, tuple)
        assert callable(fn)
        args = getargspec(fn)[0]
        assert len(inputs) == len(args)
        inputs = OrderedDict(zip(args, inputs))
        super(Function, self).__init__(inputs, output)
        self.fn = fn

    def eager_subs(self, subs):
        if not all(isinstance(subs.get(key), (Number, Tensor)) for key in self.inputs):
            return None  # defer to default implementation
        funsors = tuple(subs[key] for key in self.inputs)
        inputs, tensors = align_tensors(*funsors)
        data = self.fn(*tensors)
        inputs = None  # FIXME
        return Tensor(data, inputs, self.output)


def function(*signature):
    r"""
    Decorator to wrap PyTorch functions.

    Example::

        @funsor.function(reals(3,4), reals(4,5), reals(3,5))
        def matmul(x, y):
            return torch.matmul(x, y)

        @funsor.function(reals(3,4), reals(4,5), reals(3,5))
        def mvn_log_prob(loc, scale_tril, x):
            d = torch.distributions.MultivariateNormal(loc, scale_tril)
            return d.log_prob(x)

    :param \*signature: A sequence if input domains followed by a final output
        domain.
    """
    assert signature
    inputs, output = signature[:-1], signature[-1]
    return functools.partial(Function, inputs, output)


__all__ = [
    'Arange',
    'Function',
    'Tensor',
    'align_tensors',
    'function',
]
