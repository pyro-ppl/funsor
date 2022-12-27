# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import itertools
import numbers
import operator
from collections import OrderedDict, namedtuple
from functools import reduce

import numpy as np
import opt_einsum
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import Bint, Domain, Real
from funsor.gaussian import Gaussian
from funsor.tensor import Tensor
from funsor.terms import Funsor, Number
from funsor.util import get_backend


@contextlib.contextmanager
def xfail_if_not_implemented(msg="Not implemented", *, match=None):
    try:
        yield
    except NotImplementedError as e:
        if match is not None and match not in str(e):
            raise e from None
        import pytest

        pytest.xfail(reason="{}:\n{}".format(msg, e))


@contextlib.contextmanager
def xfail_if_not_found(msg="Not implemented"):
    try:
        yield
    except AttributeError as e:
        import pytest

        pytest.xfail(reason="{}:\n{}".format(msg, e))


def requires_backend(*backends, reason=None):
    import pytest

    if reason is None:
        reason = "Test requires backend {}".format(" or ".join(backends))
    return pytest.mark.skipif(get_backend() not in backends, reason=reason)


def excludes_backend(*backends, reason=None):
    import pytest

    if reason is None:
        reason = "Test excludes backend {}".format(" and ".join(backends))
    return pytest.mark.skipif(get_backend() in backends, reason=reason)


class ActualExpected(namedtuple("LazyComparison", ["actual", "expected"])):
    """
    Lazy string formatter for test assertions.
    """

    def __repr__(self):
        return "\n".join(["Expected:", str(self.expected), "Actual:", str(self.actual)])


def id_from_inputs(inputs):
    if isinstance(inputs, (dict, OrderedDict)):
        inputs = inputs.items()
    if not inputs:
        return "()"
    return ",".join(k + "".join(map(str, d.shape)) for k, d in inputs)


@dispatch(object, object, Variadic[float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    if type(a) != type(b):
        return False
    return ops.abs(a - b) < rtol + atol * ops.abs(b)


dispatch(np.ndarray, np.ndarray, Variadic[float])(np.allclose)


@dispatch(Tensor, Tensor, Variadic[float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    if a.inputs != b.inputs or a.output != b.output:
        return False
    return allclose(a.data, b.data, rtol=rtol, atol=atol)


def is_array(x):
    if isinstance(x, Funsor):
        return False
    if get_backend() == "torch":
        return False
    return ops.is_numeric_array(x)


def assert_close(actual, expected, atol=1e-6, rtol=1e-6):
    msg = ActualExpected(actual, expected)
    if is_array(actual):
        assert is_array(expected), msg
    elif isinstance(actual, Tensor) and is_array(actual.data):
        assert isinstance(expected, Tensor) and is_array(expected.data), msg
    elif (
        isinstance(actual, Contraction)
        and isinstance(actual.terms[0], Tensor)
        and is_array(actual.terms[0].data)
    ):
        assert isinstance(expected, Contraction) and is_array(
            expected.terms[0].data
        ), msg
    elif isinstance(actual, Contraction) and isinstance(actual.terms[0], Delta):
        assert isinstance(expected, Contraction) and isinstance(
            expected.terms[0], Delta
        ), msg
    elif isinstance(actual, Gaussian):
        assert isinstance(expected, Gaussian)
    else:
        assert type(actual) == type(expected), msg

    if isinstance(actual, Funsor):
        assert isinstance(expected, Funsor), msg
        assert actual.inputs == expected.inputs, (actual.inputs, expected.inputs)
        assert actual.output == expected.output, (actual.output, expected.output)

    if isinstance(actual, (Number, Tensor)):
        assert_close(actual.data, expected.data, atol=atol, rtol=rtol)
    elif isinstance(actual, Delta):
        assert frozenset(n for n, p in actual.terms) == frozenset(
            n for n, p in expected.terms
        )
        actual = actual.align(tuple(n for n, p in expected.terms))
        for (actual_name, (actual_point, actual_log_density)), (
            expected_name,
            (expected_point, expected_log_density),
        ) in zip(actual.terms, expected.terms):
            assert actual_name == expected_name
            assert_close(actual_point, expected_point, atol=atol, rtol=rtol)
            assert_close(actual_log_density, expected_log_density, atol=atol, rtol=rtol)
    elif isinstance(actual, Gaussian):
        # Note white_vec and prec_sqrt are expected to agree only up to an
        # orthogonal factor, but precision and info_vec should agree exactly.
        assert_close(actual._info_vec, expected._info_vec, atol=atol, rtol=rtol)
        assert_close(actual._precision, expected._precision, atol=atol, rtol=rtol)
    elif isinstance(actual, Contraction):
        assert actual.red_op == expected.red_op
        assert actual.bin_op == expected.bin_op
        assert actual.reduced_vars == expected.reduced_vars
        assert len(actual.terms) == len(expected.terms)
        for ta, te in zip(actual.terms, expected.terms):
            assert_close(ta, te, atol, rtol)
    elif type(actual).__name__ == "Tensor":
        assert get_backend() == "torch"
        import torch

        assert actual.dtype == expected.dtype, msg
        assert actual.shape == expected.shape, msg
        if actual.dtype in (torch.long, torch.uint8, torch.bool):
            assert (actual == expected).all(), msg
        else:
            eq = actual == expected
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
    elif is_array(actual):
        if get_backend() == "jax":
            import jax

            assert jax.numpy.result_type(actual.dtype) == jax.numpy.result_type(
                expected.dtype
            ), msg
        else:
            assert actual.dtype == expected.dtype, msg

        assert actual.shape == expected.shape, msg
        if actual.dtype in (np.int32, np.int64, np.uint8, bool):
            assert (actual == expected).all(), msg
        else:
            actual, expected = np.asarray(actual), np.asarray(expected)
            eq = actual == expected
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
        if actual != expected:
            diff = abs(actual - expected)
            if rtol is not None:
                assert diff < (atol + abs(expected)) * rtol, msg
            elif atol is not None:
                assert diff < atol, msg
    elif isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert set(actual) == set(expected)
        for k, actual_v in actual.items():
            assert_close(actual_v, expected[k], atol=atol, rtol=rtol)
    elif isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for actual_v, expected_v in zip(actual, expected):
            assert_close(actual_v, expected_v, atol=atol, rtol=rtol)
    else:
        raise ValueError("cannot compare objects of type {}".format(type(actual)))


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
            if get_backend() == "jax":
                # JAX has numerical errors for reducing ops.
                assert_close(x_data, data)
            else:
                assert (x_data == data).all()
        else:
            if get_backend() in ["jax", "numpy"]:
                # JAX has numerical errors for reducing ops.
                assert_close(x_data, data)
            else:
                assert x_data == data


def xfail_param(*args, **kwargs):
    import pytest

    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


def make_einsum_example(equation, fill=None, sizes=(2, 3)):
    symbols = sorted(set(equation) - set(",->"))
    sizes = {dim: size for dim, size in zip(symbols, itertools.cycle(sizes))}
    inputs, outputs = equation.split("->")
    inputs = inputs.split(",")
    outputs = outputs.split(",")
    operands = []
    for dims in inputs:
        shape = tuple(sizes[dim] for dim in dims)
        x = randn(shape)
        operand = x if fill is None else (x - x + fill)
        # no need to use pyro_dims for numpy backend
        if not isinstance(operand, np.ndarray):
            operand._pyro_dims = dims
        operands.append(operand)
    funsor_operands = [
        Tensor(operand, OrderedDict([(d, Bint[sizes[d]]) for d in inp]))
        for inp, operand in zip(inputs, operands)
    ]

    assert equation == ",".join(
        ["".join(operand.inputs.keys()) for operand in funsor_operands]
    ) + "->" + ",".join(outputs)
    return inputs, outputs, sizes, operands, funsor_operands


def assert_equiv(x, y):
    """
    Check that two funsors are equivalent up to permutation of inputs.
    """
    check_funsor(x, y.inputs, y.output, y.data)


def rand(*args):
    if isinstance(args[0], tuple):
        assert len(args) == 1
        shape = args[0]
    else:
        shape = args

    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.rand(shape)
    else:
        # work around numpy random returns float object instead of np.ndarray object when shape == ()
        return np.array(np.random.rand(*shape))


def randint(low, high, size):
    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.randint(low, high, size=size)
    else:
        return np.random.randint(low, high, size=size)


def randn(*args):
    if isinstance(args[0], tuple):
        assert len(args) == 1
        shape = args[0]
    else:
        shape = args

    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.randn(shape)
    else:
        # work around numpy random returns float object instead of np.ndarray object when shape == ()
        return np.array(np.random.randn(*shape))


def random_scale_tril(*args):
    if isinstance(args[0], tuple):
        assert len(args) == 1
        shape = args[0]
    else:
        shape = args

    from funsor.distribution import BACKEND_TO_DISTRIBUTIONS_BACKEND

    backend_dist = importlib.import_module(
        BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()]
    ).dist

    if get_backend() == "torch":
        data = randn(shape)
        return backend_dist.transforms.transform_to(
            backend_dist.constraints.lower_cholesky
        )(data)
    else:
        data = randn(shape[:-2] + (shape[-1] * (shape[-1] + 1) // 2,))
        return backend_dist.biject_to(backend_dist.constraints.lower_cholesky)(data)


def zeros(*args):
    if isinstance(args[0], tuple):
        assert len(args) == 1
        shape = args[0]
    else:
        shape = args

    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.zeros(shape)
    else:
        return np.zeros(shape)


def ones(*args):
    if isinstance(args[0], tuple):
        assert len(args) == 1
        shape = args[0]
    else:
        shape = args

    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.ones(shape)
    else:
        return np.ones(shape)


def empty(*args):
    if isinstance(args[0], tuple):
        assert len(args) == 1
        shape = args[0]
    else:
        shape = args

    backend = get_backend()
    if backend == "torch":
        import torch

        return torch.empty(shape)
    else:
        return np.empty(shape)


def random_tensor(inputs, output=Real):
    """
    Creates a random :class:`funsor.tensor.Tensor` with given inputs and output.
    """
    backend = get_backend()
    assert isinstance(inputs, OrderedDict)
    assert isinstance(output, Domain)
    shape = tuple(d.dtype for d in inputs.values()) + output.shape
    if output.dtype == "real":
        data = randn(shape)
    else:
        num_elements = reduce(operator.mul, shape, 1)
        if backend == "torch":
            import torch

            data = torch.multinomial(
                torch.ones(output.dtype), num_elements, replacement=True
            )
        else:
            data = np.random.choice(output.dtype, num_elements, replace=True)
        data = data.reshape(shape)
    return Tensor(data, inputs, output.dtype)


def random_gaussian(inputs):
    """
    Creates a random :class:`funsor.gaussian.Gaussian` with given inputs.
    """
    assert isinstance(inputs, OrderedDict)
    batch_shape = tuple(d.dtype for d in inputs.values() if d.dtype != "real")
    event_shape = (sum(d.num_elements for d in inputs.values() if d.dtype == "real"),)
    prec_sqrt = randn(batch_shape + event_shape + event_shape)
    precision = ops.matmul(prec_sqrt, ops.transpose(prec_sqrt, -1, -2))
    precision = precision + 0.5 * ops.new_eye(precision, event_shape[:1])
    prec_sqrt = ops.cholesky(precision)
    loc = randn(batch_shape + event_shape)
    white_vec = ops.matmul(prec_sqrt, ops.unsqueeze(loc, -1)).squeeze(-1)
    return Gaussian(white_vec=white_vec, prec_sqrt=prec_sqrt, inputs=inputs)


def random_mvn(batch_shape, dim, diag=False):
    """
    Generate a random :class:`torch.distributions.MultivariateNormal` with given shape.
    """
    backend = get_backend()
    rank = dim + dim
    loc = randn(batch_shape + (dim,))
    cov = randn(batch_shape + (dim, rank))
    cov = cov @ ops.transpose(cov, -1, -2)
    if diag:
        cov = cov * ops.new_eye(cov, (dim,))
    if backend == "torch":
        import pyro

        return pyro.distributions.MultivariateNormal(loc, cov)
    elif backend == "jax":
        import numpyro

        return numpyro.distributions.MultivariateNormal(loc, cov)


def make_plated_hmm_einsum(num_steps, num_obs_plates=1, num_hidden_plates=0):

    assert num_obs_plates >= num_hidden_plates
    t0 = num_obs_plates + 1

    obs_plates = "".join(opt_einsum.get_symbol(i) for i in range(num_obs_plates))
    hidden_plates = "".join(opt_einsum.get_symbol(i) for i in range(num_hidden_plates))

    inputs = [str(opt_einsum.get_symbol(t0))]
    for t in range(t0, num_steps + t0):
        inputs.append(
            str(opt_einsum.get_symbol(t))
            + str(opt_einsum.get_symbol(t + 1))
            + hidden_plates
        )
        inputs.append(str(opt_einsum.get_symbol(t + 1)) + obs_plates)
    equation = ",".join(inputs) + "->"
    return (equation, "".join(sorted(tuple(set(obs_plates + hidden_plates)))))


def make_chain_einsum(num_steps):
    inputs = [str(opt_einsum.get_symbol(0))]
    for t in range(num_steps):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t + 1)))
    equation = ",".join(inputs) + "->"
    return equation


def make_hmm_einsum(num_steps):
    inputs = [str(opt_einsum.get_symbol(0))]
    for t in range(num_steps):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t + 1)))
        inputs.append(str(opt_einsum.get_symbol(t + 1)))
    equation = ",".join(inputs) + "->"
    return equation


def iter_subsets(iterable, *, min_size=None, max_size=None):
    if min_size is None:
        min_size = 0
    if max_size is None:
        max_size = len(iterable)
    for size in range(min_size, max_size + 1):
        yield from itertools.combinations(iterable, size)


class DesugarGetitem:
    """
    Helper to desugar ``.__getitem__()`` syntax.

    Example::

        >>> desugar_getitem[1:3, ..., None]
        (slice(1, 3), Ellipsis, None)
    """

    def __getitem__(self, index):
        return index


desugar_getitem = DesugarGetitem()
