from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import torch
from six import add_metaclass

import funsor.ops as ops
from funsor.domains import Domain, ints
from funsor.six import getargspec
from funsor.terms import Binary, Funsor, FunsorMeta, Number, Variable, to_funsor, eager


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
    # Collect nominal shapes.
    inputs = OrderedDict()
    for x in args:
        inputs.update(x.inputs)

    # Compute linear shapes.
    sizes = []
    shapes = []
    for domain in inputs.values():
        size = domain.dtype
        assert isinstance(size, int)
        sizes.append(size ** domain.num_elements)
        shapes.append((size,) * domain.num_elements)
    sizes = dict(zip(inputs, sizes))
    shapes = dict(zip(inputs, shapes))

    # Convert each Number or Tensor.
    tensors = []
    for i, x in enumerate(args):
        if isinstance(x, Number):
            tensors.append(x)
            continue

        x_inputs, x_output, x = x.inputs, x.output, x.data
        if x_inputs == inputs:
            tensors.append(x)
            continue

        # Squash each multivariate input dim into a single dim.
        x = x.reshape(tuple(sizes[k] for k in x_inputs) + x_output.shape)

        # Pemute squashed input dims.
        x_keys = tuple(x_inputs)
        x = x.data.permute(tuple(x_keys.index(k) for k in inputs if k in x_inputs))

        # Fill in ones.
        x = x.reshape(tuple(sizes[k] if k in x_inputs else 1 for k in inputs) +
                      x_output.shape)

        # Unsquash multivariate input dims.
        x = x.reshape(sum((shapes[k] if k in x_inputs else (1,) * len(shapes[k]) for k in inputs),
                          ()) + x_output.shape)
        tensors.append(x)

    return inputs, tensors


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
        input_dim = sum(i.num_elements for i in inputs.values())
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
        assert isinstance(subs, tuple)
        if all(isinstance(v, (Number, Tensor)) and not v.inputs or
                isinstance(v, Variable)
                for k, v in subs):
            # Substitute one variable at a time.
            conflicts = [k for k, v in subs]
            result = self
            for i, (key, value) in enumerate(subs):
                if isinstance(value, Funsor):
                    if not set(value.inputs).isdisjoint(conflicts[1 + i:]):
                        raise NotImplementedError(
                            'TODO implement simultaneous substitution')
                result = result._eager_subs_one(key, value)
            return result

        return None  # defer to default implementation

    def _eager_subs_one(self, name, value):
        if isinstance(value, Variable):
            # Simply rename an input.
            inputs = OrderedDict((value.name if k == name else k, v)
                                 for k, v in self.inputs.items())
            return Tensor(self.data, inputs, self.dtype)

        if isinstance(value, Number):
            assert self.inputs[name].num_elements == 1
            inputs = OrderedDict((k, v) for k, v in self.inputs.items() if k != name)
            index = []
            for k, domain in self.inputs.items():
                if k == name:
                    index.append(int(value.data))
                    break
                else:
                    index.extend([slice(None)] * domain.num_elements)
            data = self.data[tuple(index)]
            return Tensor(data, inputs, self.dtype)

        if isinstance(value, Tensor):
            raise NotImplementedError('TODO')

        raise RuntimeError('{} should be handled by caller'.format(value))

    def eager_unary(self, op):
        return Tensor(op(self.data), self.inputs, self.dtype)

    def eager_reduce(self, op, reduced_vars):
        if op in ops.REDUCE_OP_TO_TORCH:
            torch_op = ops.REDUCE_OP_TO_TORCH[op]
            assert isinstance(reduced_vars, frozenset)
            self_vars = frozenset(self.inputs)
            reduced_vars = reduced_vars & self_vars
            if reduced_vars == self_vars:
                # Reduce all dims at once.
                if op is ops.logaddexp:
                    # work around missing torch.Tensor.logsumexp()
                    data = self.data.reshape(-1).logsumexp(0)
                    return Tensor(data, dtype=self.dtype)
                return Tensor(torch_op(self.data), dtype=self.dtype)

            # Reduce one dim at a time.
            data = self.data
            offset = 0
            for k, domain in self.inputs.items():
                if k in reduced_vars:
                    if domain.num_elements > 1:
                        raise NotImplementedError('TODO')
                    data = torch_op(data, dim=offset)
                    if op is ops.min or op is ops.max:
                        data = data[0]
                else:
                    offset += domain.num_elements
            inputs = OrderedDict((k, v) for k, v in self.inputs.items()
                                 if k not in reduced_vars)
            return Tensor(data, inputs, self.dtype)
        return super(Tensor, self).eager_reduce(op, reduced_vars)


@eager.register(Binary, object, Tensor, Number)
def eager_binary_tensor_number(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    return Tensor(data, lhs.inputs, lhs.dtype)


@eager.register(Binary, object, Number, Tensor)
def eager_binary_number_tensor(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    return Tensor(data, rhs.inputs, rhs.dtype)


@eager.register(Binary, object, Tensor, Tensor)
def eager_binary_tensor_tensor(op, lhs, rhs):
    assert lhs.dtype == rhs.dtype
    if lhs.inputs == rhs.inputs:
        inputs = lhs.inputs
        data = op(lhs.data, rhs.data)
    else:
        inputs, (lhs_data, rhs_data) = align_tensors(lhs, rhs)
        data = op(lhs_data, rhs_data)
    return Tensor(data, inputs, lhs.dtype)


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


class Function(Funsor):
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
        assert isinstance(subs, tuple)
        subs = dict(subs)
        if not all(isinstance(subs.get(key), (Number, Tensor)) for key in self.inputs):
            return None  # defer to default implementation

        funsors = tuple(subs[key] for key in self.inputs)
        inputs, tensors = align_tensors(*funsors)
        data = self.fn(*tensors)
        inputs = None  # FIXME
        return Tensor(data, inputs, self.dtype)


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
