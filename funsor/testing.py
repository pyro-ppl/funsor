# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import itertools
import numbers
import operator
from collections import OrderedDict, namedtuple
from functools import reduce

import numpy as np
import opt_einsum
import pytest
import torch
from jax.dtypes import canonicalize_dtype
from jax.interpreters.xla import DeviceArray
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import Domain, bint, reals
from funsor.gaussian import Gaussian
from funsor.numpy import array
from funsor.terms import Funsor, Number
from funsor.tensor import Tensor


@contextlib.contextmanager
def xfail_if_not_implemented(msg="Not implemented"):
    try:
        yield
    except NotImplementedError as e:
        pytest.xfail(reason='{}:\n{}'.format(msg, e))


class ActualExpected(namedtuple('LazyComparison', ['actual', 'expected'])):
    """
    Lazy string formatter for test assertions.
    """
    def __repr__(self):
        return '\n'.join(['Expected:', str(self.expected), 'Actual:', str(self.actual)])


def id_from_inputs(inputs):
    if isinstance(inputs, (dict, OrderedDict)):
        inputs = inputs.items()
    if not inputs:
        return '()'
    return ','.join(k + ''.join(map(str, d.shape)) for k, d in inputs)


@dispatch(object, object, Variadic[float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    if type(a) != type(b):
        return False
    return ops.abs(a - b) < rtol + atol * ops.abs(b)


dispatch(np.ndarray, np.ndarray, Variadic[float])(np.allclose)


@dispatch(torch.Tensor, torch.Tensor, Variadic[float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


@dispatch(Tensor, Tensor, Variadic[float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    if a.inputs != b.inputs or a.output != b.output:
        return False
    return allclose(a.data, b.data, rtol=rtol, atol=atol)


def assert_close(actual, expected, atol=1e-6, rtol=1e-6):
    msg = ActualExpected(actual, expected)
    if isinstance(actual, DeviceArray):
        assert isinstance(expected, array), msg
    else:
        assert type(actual) == type(expected), msg
    if isinstance(actual, Funsor):
        assert isinstance(actual, Funsor)
        assert isinstance(expected, Funsor)
        assert actual.inputs == expected.inputs, (actual.inputs, expected.inputs)
        assert actual.output == expected.output, (actual.output, expected.output)

    if isinstance(actual, (Number, Tensor)):
        assert_close(actual.data, expected.data, atol=atol, rtol=rtol)
    elif isinstance(actual, Delta):
        assert frozenset(n for n, p in actual.terms) == frozenset(n for n, p in expected.terms)
        actual = actual.align(tuple(n for n, p in expected.terms))
        for (actual_name, (actual_point, actual_log_density)), \
                (expected_name, (expected_point, expected_log_density)) in \
                zip(actual.terms, expected.terms):
            assert actual_name == expected_name
            assert_close(actual_point, expected_point, atol=atol, rtol=rtol)
            assert_close(actual_log_density, expected_log_density, atol=atol, rtol=rtol)
    elif isinstance(actual, Gaussian):
        assert_close(actual.info_vec, expected.info_vec, atol=atol, rtol=rtol)
        assert_close(actual.precision, expected.precision, atol=atol, rtol=rtol)
    elif isinstance(actual, Contraction):
        assert actual.red_op == expected.red_op
        assert actual.bin_op == expected.bin_op
        assert actual.reduced_vars == expected.reduced_vars
        assert len(actual.terms) == len(expected.terms)
        for ta, te in zip(actual.terms, expected.terms):
            assert_close(ta, te, atol, rtol)
    elif isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype, msg
        assert actual.shape == expected.shape, msg
        if actual.dtype in (torch.long, torch.uint8, torch.bool):
            assert (actual == expected).all(), msg
        else:
            eq = (actual == expected)
            if eq.all():
                return
            if eq.any():
                actual = actual[~eq]
                expected = expected[~eq]
            diff = (actual.detach() - expected.detach()).abs()
            if rtol is not None:
                assert (diff / (atol + expected.detach().abs())).max() < rtol, msg
            elif atol is not None:
                assert diff.max() < atol, msg
    elif isinstance(actual, array):
        assert actual.dtype == canonicalize_dtype(expected.dtype), msg
        assert actual.shape == expected.shape, msg
        if actual.dtype in (np.int32, np.int64, np.uint8, np.bool):
            assert (actual == expected).all(), msg
        else:
            eq = (actual == expected)
            if eq.all():
                return
            if eq.any():
                actual = actual[~eq]
                expected = expected[~eq]
            diff = abs(actual - expected)
            if rtol is not None:
                assert (diff / (atol + abs(expected))).max() < rtol, msg
            elif atol is not None:
                assert diff.max() < atol, msg
    elif isinstance(actual, numbers.Number):
        diff = abs(actual - expected)
        if rtol is not None:
            assert diff < (atol + abs(expected)) * rtol, msg
        elif atol is not None:
            assert diff < atol, msg
    else:
        raise ValueError('cannot compare objects of type {}'.format(type(actual)))


def check_funsor(x, inputs, output, data=None):
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, Funsor)
    assert dict(x.inputs) == dict(inputs)
    if output is not None:
        assert x.output == output
    if data is not None:
        if x.inputs == inputs:
            x_data = x.data
        else:
            x_data = x.align(tuple(inputs)).data
        if inputs or output.shape:
            assert (x_data == data).all()
        else:
            assert x_data == data


def xfail_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


def make_einsum_example(equation, fill=None, sizes=(2, 3)):
    symbols = sorted(set(equation) - set(',->'))
    sizes = {dim: size for dim, size in zip(symbols, itertools.cycle(sizes))}
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    outputs = outputs.split(',')
    operands = []
    for dims in inputs:
        shape = tuple(sizes[dim] for dim in dims)
        operands.append(torch.randn(shape) if fill is None else torch.full(shape, fill))
        operands[-1]._pyro_dims = dims
    funsor_operands = [
        Tensor(operand, OrderedDict([(d, bint(sizes[d])) for d in inp]))
        for inp, operand in zip(inputs, operands)
    ]

    assert equation == \
        ",".join(["".join(operand.inputs.keys()) for operand in funsor_operands]) + "->" + ",".join(outputs)
    return inputs, outputs, sizes, operands, funsor_operands


def assert_equiv(x, y):
    """
    Check that two funsors are equivalent up to permutation of inputs.
    """
    check_funsor(x, y.inputs, y.output, y.data)


def rand(shape, backend="torch"):
    assert backend in ["torch", "numpy"]
    if backend == "torch":
        return torch.rand(shape)
    else:
        # work around numpy random returns float object instead of np.ndarray object when shape == ()
        return np.array(np.random.rand(*shape))


def randn(shape, backend="torch"):
    assert backend in ["torch", "numpy"]
    if backend == "torch":
        return torch.randn(shape)
    else:
        # work around numpy random returns float object instead of np.ndarray object when shape == ()
        return np.array(np.random.randn(*shape))


def astype(x, dtype):
    if torch.is_tensor(x):
        if dtype == 'uint8':
            return x.byte()
        return x.type(dtype)
    else:
        return x.astype(dtype)


def random_tensor(inputs, output=reals(), backend="torch"):
    """
    Creates a random :class:`funsor.tensor.Tensor` with given inputs and output.
    """
    assert isinstance(inputs, OrderedDict)
    assert isinstance(output, Domain)
    shape = tuple(d.dtype for d in inputs.values()) + output.shape
    if output.dtype == 'real':
        data = torch.randn(shape) if backend == "torch" else np.array(np.random.randn(*shape))
    else:
        num_elements = reduce(operator.mul, shape, 1)
        if backend == "torch":
            data = torch.multinomial(torch.ones(output.dtype), num_elements, replacement=True)
        else:
            data = np.random.choice(output.dtype, num_elements, replace=True)
        data = data.reshape(shape)
    return Tensor(data, inputs, output.dtype)


def random_gaussian(inputs, backend="torch"):
    """
    Creates a random :class:`funsor.gaussian.Gaussian` with given inputs.
    """
    assert isinstance(inputs, OrderedDict)
    batch_shape = tuple(d.dtype for d in inputs.values() if d.dtype != 'real')
    event_shape = (sum(d.num_elements for d in inputs.values() if d.dtype == 'real'),)
    prec_sqrt = randn(batch_shape + event_shape + event_shape, backend)
    precision = ops.matmul(prec_sqrt, ops.transpose(prec_sqrt, -1, -2))
    precision = precision + 0.5 * ops.new_eye(precision, event_shape[:1])
    loc = randn(batch_shape + event_shape, backend)
    info_vec = ops.matmul(precision, ops.unsqueeze(loc, -1)).squeeze(-1)
    return Gaussian(info_vec, precision, inputs)


def random_mvn(batch_shape, dim, diag=False):
    """
    Generate a random :class:`torch.distributions.MultivariateNormal` with given shape.
    """
    rank = dim + dim
    loc = torch.randn(batch_shape + (dim,))
    cov = torch.randn(batch_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    if diag:
        cov = cov * torch.eye(dim)
    return torch.distributions.MultivariateNormal(loc, cov)


def make_plated_hmm_einsum(num_steps, num_obs_plates=1, num_hidden_plates=0):

    assert num_obs_plates >= num_hidden_plates
    t0 = num_obs_plates + 1

    obs_plates = ''.join(opt_einsum.get_symbol(i) for i in range(num_obs_plates))
    hidden_plates = ''.join(opt_einsum.get_symbol(i) for i in range(num_hidden_plates))

    inputs = [str(opt_einsum.get_symbol(t0))]
    for t in range(t0, num_steps+t0):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t+1)) + hidden_plates)
        inputs.append(str(opt_einsum.get_symbol(t+1)) + obs_plates)
    equation = ",".join(inputs) + "->"
    return (equation, ''.join(sorted(tuple(set(obs_plates + hidden_plates)))))


def make_chain_einsum(num_steps):
    inputs = [str(opt_einsum.get_symbol(0))]
    for t in range(num_steps):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t+1)))
    equation = ",".join(inputs) + "->"
    return equation


def make_hmm_einsum(num_steps):
    inputs = [str(opt_einsum.get_symbol(0))]
    for t in range(num_steps):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t+1)))
        inputs.append(str(opt_einsum.get_symbol(t+1)))
    equation = ",".join(inputs) + "->"
    return equation
