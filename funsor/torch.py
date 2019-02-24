from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import torch
from six import add_metaclass, integer_types

import funsor.ops as ops
from funsor.domains import Domain, ints
from funsor.six import getargspec
from funsor.terms import Binary, Funsor, FunsorMeta, Number, Variable, eager, to_funsor


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
        if not any(k in self.inputs for k, v in subs):
            return self
        for k, v in subs:
            if k in self.inputs:
                for name, domain in v.inputs.items():
                    if not isinstance(domain.dtype, integer_types):
                        raise ValueError('Tensors can only depend on integer free variables, '
                                         'but substituting would make this tensor depend on '
                                         '"{}" of domain {}'.format(name, domain))

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

        raise NotImplementedError('TODO support advanced indexing into Tensor')

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

        raise NotImplementedError('TODO support advanced indexing into Tensor')

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
    if op is ops.getitem:
        # Shift by that Funsor is using for inputs.
        index = [slice(None)] * sum(d.num_elements for d in lhs.inputs.values())
        index.append(rhs.data)
        index = tuple(index)
        data = lhs.data[index]
    else:
        data = op(lhs.data, rhs.data)
    return Tensor(data, lhs.inputs, lhs.dtype)


@eager.register(Binary, object, Number, Tensor)
def eager_binary_number_tensor(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    return Tensor(data, rhs.inputs, rhs.dtype)


@eager.register(Binary, object, Tensor, Tensor)
def eager_binary_tensor_tensor(op, lhs, rhs):
    assert lhs.dtype == rhs.dtype
    if op is ops.getitem:
        raise NotImplementedError('TODO shift dim to index on')
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
    r"""
    Funsor wrapped by a PyTorch function.

    Functions are support broadcasting and can be eagerly evaluated on funsors
    with free variables of int type (i.e. batch dimensions).

    :class:`Function`s are often created via the :func:`function` decorator.

    :param callable fn: A PyTorch function to wrap.
    :param funsor.domains.Domain output: An output domain.
    :param Funsor \*args: Funsor arguments.
    """
    def __init__(self, fn, output, args):
        assert callable(fn)
        assert isinstance(args, tuple)
        inputs = OrderedDict()
        for arg in args:
            assert isinstance(arg, Funsor)
            inputs.update(arg.inputs)
        super(Function, self).__init__(inputs, output)
        self.fn = fn
        self.args = args

    def __repr__(self):
        return 'Function({})'.format(', '.join(
            [type(self).__name__, repr(self.output)] + list(map(repr, self.args))))

    def __str__(self):
        return 'Function({})'.format(', '.join(
            [type(self).__name__, str(self.output)] + list(map(str, self.args))))

    def eager_subs(self, subs):
        if not any(k in self.inputs for k, v in subs):
            return self
        args = tuple(arg.eager_subs(subs) for arg in self.args)
        return Function(self.fn, self.output, args)


@eager.register(Function, object, Domain, tuple)
def eager_function(fn, output, args):
    if not all(isinstance(arg, (Number, Tensor)) for arg in args):
        return None  # defer to default implementation
    inputs, tensors = align_tensors(*args)
    data = fn(*tensors)
    result = Tensor(data, inputs, dtype=output.dtype)
    assert result.output == output
    return result


def _function(inputs, output, fn):
    names = getargspec(fn)[0]
    args = tuple(Variable(name, domain) for (name, domain) in zip(names, inputs))
    assert len(args) == len(inputs)
    return Function(fn, output, args)


def function(*signature):
    r"""
    Decorator to wrap a PyTorch function.

    Example::

        @funsor.function(reals(3,4), reals(4,5), reals(3,5))
        def matmul(x, y):
            return torch.matmul(x, y)

        @funsor.function(reals(10), reals(10, 10), reals())
        def mvn_log_prob(loc, scale_tril, x):
            d = torch.distributions.MultivariateNormal(loc, scale_tril)
            return d.log_prob(x)

    :param \*signature: A sequence if input domains followed by a final output
        domain.
    """
    assert signature
    assert all(isinstance(d, Domain) for d in signature)
    inputs, output = signature[:-1], signature[-1]
    return functools.partial(_function, inputs, output)


__all__ = [
    'Arange',
    'Function',
    'Tensor',
    'align_tensors',
    'function',
]
