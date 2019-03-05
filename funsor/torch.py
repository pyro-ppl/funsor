from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import torch
from six import add_metaclass, integer_types

import funsor.ops as ops
from funsor.domains import Domain, bint, find_domain, reals
from funsor.six import getargspec
from funsor.terms import Binary, Funsor, FunsorMeta, Number, Variable, eager, to_funsor


def align_tensor(new_inputs, x):
    r"""
    Permute and expand a tensor to match desired ``new_inputs``.

    :param OrderedDict new_inputs: A target set of inputs.
    :param funsor.terms.Funsor x: A :class:`Tensor`s or
        :class:`~funsor.terms.Number`.
    :return: a number or :class:`torch.Tensor` that can be broadcast to other
        tensors with inputs ``new_inputs``.
    :rtype: tuple
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(x, (Number, Tensor))
    assert all(isinstance(d.dtype, integer_types) for d in x.inputs.values())

    data = x.data
    if isinstance(x, Number):
        return data

    old_inputs = x.inputs
    if old_inputs == new_inputs:
        return data

    # Permute squashed input dims.
    x_keys = tuple(old_inputs)
    data = data.permute(tuple(x_keys.index(k) for k in new_inputs if k in old_inputs) +
                        tuple(range(len(old_inputs), data.dim())))

    # Unsquash multivariate input dims by filling in ones.
    data = data.reshape(tuple(old_inputs[k].dtype if k in old_inputs else 1 for k in new_inputs) +
                        x.output.shape)
    return data


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
    inputs = OrderedDict()
    for x in args:
        inputs.update(x.inputs)
    tensors = [align_tensor(inputs, x) for x in args]
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
        assert all(isinstance(d.dtype, integer_types) for k, d in inputs)
        inputs = OrderedDict(inputs)
        output = Domain(data.shape[len(inputs):], dtype)
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

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self
        inputs = OrderedDict((name, self.inputs[name]) for name in names)
        inputs.update(self.inputs)

        if any(d.shape for d in self.inputs.values()):
            raise NotImplementedError("TODO: Implement align with vector indices.")
        old_dims = tuple(self.inputs)
        new_dims = tuple(inputs)
        data = self.data.permute(tuple(old_dims.index(d) for d in new_dims))
        return Tensor(data, inputs, self.dtype)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = {k: materialize(v) for k, v in subs if k in self.inputs}
        if not subs:
            return self

        # Compute result shapes.
        inputs = OrderedDict()
        for k, domain in self.inputs.items():
            if k in subs:
                inputs.update(subs[k].inputs)
            else:
                inputs[k] = domain

        # Construct a dict with each input's positional dim,
        # counting from the right so as to support broadcasting.
        total_size = len(inputs) + len(self.output.shape)  # Assumes only scalar indices.
        new_dims = {}
        for k, domain in inputs.items():
            assert not domain.shape
            new_dims[k] = len(new_dims) - total_size

        # Use advanced indexing to construct a simultaneous substitution.
        index = []
        for k, domain in self.inputs.items():
            if k in subs:
                v = subs.get(k)
                if isinstance(v, Number):
                    index.append(int(v.data))
                else:
                    # Permute and expand v.data to end up at new_dims.
                    assert isinstance(v, Tensor)
                    v = v.align(tuple(k2 for k2 in inputs if k2 in v.inputs))
                    assert isinstance(v, Tensor)
                    v_shape = [1] * total_size
                    for k2, size in zip(v.inputs, v.data.shape):
                        v_shape[new_dims[k2]] = size
                    index.append(v.data.reshape(tuple(v_shape)))
            else:
                # Construct a [:] slice for this preserved input.
                offset_from_right = -1 - new_dims[k]
                index.append(torch.arange(domain.dtype).reshape(
                    (-1,) + (1,) * offset_from_right))

        # Construct a [:] slice for the output.
        for i, size in enumerate(self.output.shape):
            offset_from_right = len(self.output.shape) - i - 1
            index.append(torch.arange(size).reshape(
                (-1,) + (1,) * offset_from_right))

        data = self.data[tuple(index)]
        return Tensor(data, inputs, self.dtype)

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
                    assert not domain.shape
                    data = torch_op(data, dim=offset)
                    if op is ops.min or op is ops.max:
                        data = data[0]
                else:
                    offset += 1
            inputs = OrderedDict((k, v) for k, v in self.inputs.items()
                                 if k not in reduced_vars)
            return Tensor(data, inputs, self.dtype)
        return super(Tensor, self).eager_reduce(op, reduced_vars)


@eager.register(Binary, object, Tensor, Number)
def eager_binary_tensor_number(op, lhs, rhs):
    if op is ops.getitem:
        # Shift by that Funsor is using for inputs.
        index = [slice(None)] * len(lhs.inputs)
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
    # Compute inputs and outputs.
    dtype = find_domain(op, lhs.output, rhs.output).dtype
    if lhs.inputs == rhs.inputs:
        inputs = lhs.inputs
        lhs_data, rhs_data = lhs.data, rhs.data
    else:
        inputs, (lhs_data, rhs_data) = align_tensors(lhs, rhs)

    if op is ops.getitem:
        # getitem has special shape semantics.
        if rhs.output.shape:
            raise NotImplementedError('TODO support vector indexing')
        assert lhs.output.shape == (rhs.dtype,)
        index = [torch.arange(size).reshape((-1,) + (1,) * (lhs_data.dim() - pos - 2))
                 for pos, size in enumerate(lhs_data.shape)]
        index[-1] = rhs_data
        data = lhs_data[tuple(index)]
    else:
        data = op(lhs_data, rhs_data)

    return Tensor(data, inputs, dtype)


def arange(name, size):
    """
    Helper to create a named :func:`torch.arange` funsor.

    :param str name: A variable name.
    :param int size: A size.
    :rtype: Tensor
    """
    data = torch.arange(size)
    inputs = OrderedDict([(name, bint(size))])
    return Tensor(data, inputs, dtype=size)


def materialize(x):
    """
    Attempt to convert a Funsor to a :class:`~funsor.terms.Number` or
    :class:`Tensor` by substituting :func:`arange`s into its free variables.
    """
    assert isinstance(x, Funsor)
    if isinstance(x, (Number, Tensor)):
        return x
    subs = []
    for name, domain in x.inputs.items():
        if not isinstance(domain.dtype, integer_types):
            raise ValueError('materialize() requires integer free variables but found '
                             '"{}" of domain {}'.format(name, domain))
        assert not domain.shape
        subs.append((name, arange(name, domain.dtype)))
    subs = tuple(subs)
    return x.eager_subs(subs)


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


def torch_einsum(equation, *operands):
    """
    Wrapper around :func:`torch.einsum` to operate on real-valued Funsors.

    Note this operates only on the ``output`` tensor. To perform sum-product
    contractions on named dimensions, instead use ``+`` and
    :class:`~funsor.terms.Reduce`.
    """
    assert isinstance(equation, str)
    assert isinstance(operands, tuple)
    for x in operands:
        assert isinstance(x, Funsor)
        assert x.dtype == 'real'
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    sizes = {dim: size
             for input_, operand in zip(inputs, operands)
             for dim, size in zip(input_, operand.output.shape)}
    output = reals(*(sizes[dim] for dim in output))
    fn = functools.partial(torch.einsum, equation)
    return Function(fn, output, operands)


__all__ = [
    'Function',
    'Tensor',
    'align_tensor',
    'align_tensors',
    'arange',
    'torch_einsum',
    'function',
    'materialize',
]
