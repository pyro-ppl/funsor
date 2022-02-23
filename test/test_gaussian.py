# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools
import pprint
from collections import OrderedDict
from functools import reduce

import numpy as np
import pytest

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.domains import Bint, Real, Reals
from funsor.gaussian import (
    BlockMatrix,
    BlockVector,
    Gaussian,
    _compress_rank,
    _inverse_cholesky,
    _norm2,
    _split_real_inputs,
    _vm,
)
from funsor.integrate import Integrate
from funsor.interpretations import eager, lazy
from funsor.montecarlo import extract_samples
from funsor.tensor import Einsum, Tensor, numeric_array
from funsor.terms import Number, Subs, Unary, Variable
from funsor.testing import (
    assert_close,
    id_from_inputs,
    ones,
    randn,
    random_gaussian,
    random_tensor,
    requires_backend,
    zeros,
)
from funsor.util import get_backend

assert Einsum  # flake8


@requires_backend("torch")
@pytest.mark.parametrize("size", [1, 2, 3], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
def test_cholesky_solve(batch_shape, size):
    import torch

    b = torch.randn(batch_shape + (size, 5))
    x = torch.randn(batch_shape + (size, size))
    x = x.transpose(-1, -2).matmul(x)
    u = x.cholesky()
    expected = torch.cholesky_solve(b, u)
    assert not expected.requires_grad
    actual = torch.cholesky_solve(b.requires_grad_(), u.requires_grad_())
    assert actual.requires_grad
    assert_close(expected, actual)


def naive_cholesky_inverse(u):
    import torch

    shape = u.shape
    return torch.stack(
        [part.cholesky_inverse() for part in u.reshape((-1,) + u.shape[-2:])]
    ).reshape(shape)


@requires_backend("torch")
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
def test_cholesky_inverse(batch_shape, size, requires_grad):
    import torch

    x = torch.randn(batch_shape + (size, size))
    x = x.transpose(-1, -2).matmul(x)
    u = x.cholesky()
    if requires_grad:
        u.requires_grad_()
    assert_close(ops.cholesky_inverse(u), naive_cholesky_inverse(u))
    if requires_grad:
        ops.cholesky_inverse(u).sum().backward()


@pytest.mark.parametrize("size", [1, 2, 3], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
def test_inverse_cholesky(batch_shape, size):
    prec_sqrt = randn((size, size))
    precision = prec_sqrt @ ops.transpose(prec_sqrt, -1, -2)

    # The naive computation requires two Choleskys + two triangular_solves.
    precision_chol = ops.cholesky(precision)
    covariance = ops.cholesky_inverse(precision_chol)
    expected = ops.cholesky(covariance)

    # Du's trick requires only a single Cholesky + single triangular_solve.
    actual = _inverse_cholesky(precision)
    assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_split_real_inputs():
    inputs = OrderedDict(i=Bint[5], a=Real, b=Reals[4], c=Reals[3, 1], d=Reals[1, 2])
    prototype = randn(())
    g = random_gaussian(inputs)
    for lhs_keys in "a b c d ab ac ad bc bd abc abd acd bcd".split():
        a, b = _split_real_inputs(inputs, lhs_keys, prototype)
        prec_sqrt_a = g.prec_sqrt[..., a, :]
        prec_sqrt_b = g.prec_sqrt[..., b, :]
        assert prec_sqrt_a.shape[-2] == sum(
            d.num_elements for k, d in inputs.items() if k in lhs_keys
        )
        assert prec_sqrt_a.shape[-2] == sum(
            d.num_elements for k, d in inputs.items() if k in lhs_keys
        )
        prec_sqrt_ab = ops.cat([prec_sqrt_a, prec_sqrt_b], -2)
        assert prec_sqrt_ab.shape == g.prec_sqrt.shape


def test_block_vector():
    shape = (10,)
    expected = zeros(shape)
    actual = BlockVector(shape)

    expected[1] = randn(())
    actual[1] = expected[1]

    expected[3:5] = randn((2,))
    actual[3:5] = expected[3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)])
def test_block_vector_batched(batch_shape):
    shape = batch_shape + (10,)
    expected = zeros(shape)
    actual = BlockVector(shape)

    expected[..., 1] = randn(batch_shape)
    actual[..., 1] = expected[..., 1]

    expected[..., 3:5] = randn(batch_shape + (2,))
    actual[..., 3:5] = expected[..., 3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_block_matrix(sparse):
    shape = (10, 10)
    expected = zeros(shape)
    actual = BlockMatrix(shape)

    expected[1, 1] = randn(())
    actual[1, 1] = expected[1, 1]

    if not sparse:
        expected[1, 3:5] = randn((2,))
        actual[1, 3:5] = expected[1, 3:5]

        expected[3:5, 1] = randn((2,))
        actual[3:5, 1] = expected[3:5, 1]

    expected[3:5, 3:5] = randn((2, 2))
    actual[3:5, 3:5] = expected[3:5, 3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)])
def test_block_matrix_batched(batch_shape, sparse):
    shape = batch_shape + (10, 10)
    expected = zeros(shape)
    actual = BlockMatrix(shape)

    expected[..., 1, 1] = randn(batch_shape)
    actual[..., 1, 1] = expected[..., 1, 1]

    if not sparse:
        expected[..., 1, 3:5] = randn(batch_shape + (2,))
        actual[..., 1, 3:5] = expected[..., 1, 3:5]

        expected[..., 3:5, 1] = randn(batch_shape + (2,))
        actual[..., 3:5, 1] = expected[..., 3:5, 1]

    expected[..., 3:5, 3:5] = randn(batch_shape + (2, 2))
    actual[..., 3:5, 3:5] = expected[..., 3:5, 3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize("batch_shape", [(), (3, 2), (4,)], ids=str)
@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, 8, 13])
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("method", ["cholesky", "qr"])
def test_compress_rank(batch_shape, dim, rank, method):
    white_vec = randn(batch_shape + (rank,))
    prec_sqrt = randn(batch_shape + (dim, rank))

    shift = zeros(batch_shape)
    new_white_vec = white_vec
    new_prec_sqrt = prec_sqrt
    if rank >= dim:
        new_white_vec, new_prec_sqrt, shift = _compress_rank(
            white_vec, prec_sqrt, method
        )
    assert new_prec_sqrt.shape[:-1] == batch_shape + (dim,)
    assert new_white_vec.shape[:-1] == batch_shape
    assert new_prec_sqrt.shape[-1] == new_white_vec.shape[-1]
    assert shift.shape == batch_shape
    new_rank = new_prec_sqrt.shape[-1]
    assert new_rank <= dim

    # Check precisions.
    expected_precision = prec_sqrt @ ops.transpose(prec_sqrt, -1, -2)
    actual_precision = new_prec_sqrt @ ops.transpose(new_prec_sqrt, -1, -2)
    assert_close(actual_precision, expected_precision, atol=1e-5, rtol=None)

    # Check full evaluation.
    probe = randn(batch_shape + (dim,))
    expected = -0.5 * _norm2(_vm(probe, prec_sqrt) - white_vec)
    actual = -0.5 * _norm2(_vm(probe, new_prec_sqrt) - new_white_vec) + shift
    assert_close(actual, expected, atol=1e-4, rtol=None)


@pytest.mark.parametrize("batch_shape", [(), (3, 2), (4,)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
def test_compress_rank_singular(batch_shape, dim):
    rank = dim + 1
    white_vec = zeros(batch_shape + (rank,))
    prec_sqrt = zeros(batch_shape + (dim, rank))

    # Check that _compress_rank can handle singular prec_sqrt.
    white_vec, prec_sqrt, shift = _compress_rank(white_vec, prec_sqrt)
    assert_close(white_vec, zeros(batch_shape + (dim,)))
    assert_close(prec_sqrt, zeros(batch_shape + (dim, dim)))


@pytest.mark.parametrize(
    "dim, rank", [(d, r) for d in range(1, 6) for r in range(1, 21) if d < r]
)
def test_compress_rank_gaussian(dim, rank):
    inputs = OrderedDict(x=Reals[dim])
    white_vec = randn((rank,))
    prec_sqrt = randn((dim, rank))
    data = randn((dim,))
    with Gaussian.set_compression_threshold(999):
        g1 = Gaussian(white_vec, prec_sqrt, inputs)
        g1_data = g1(x=data)
    assert isinstance(g1, Gaussian)
    assert g1.rank == rank

    white_vec, prec_sqrt, shift = _compress_rank(white_vec, prec_sqrt)
    assert white_vec.shape == (dim,)
    assert prec_sqrt.shape == (dim, dim)
    g2 = Gaussian(white_vec, prec_sqrt, inputs)
    g2_data = g2(x=data)
    assert isinstance(g2, Gaussian)
    assert g2.rank == dim

    assert_close(g1._mean, g2._mean, atol=1e-4, rtol=1e-3)
    assert_close(g1._precision, g2._precision, atol=1e-4, rtol=1e-3)

    actual = g1.reduce(ops.logaddexp)
    expected = g2.reduce(ops.logaddexp) + shift
    assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    actual = g1_data
    expected = g2_data + shift
    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "loc, scale",
    [
        ("white_vec", "prec_sqrt"),
        ("_mean", "prec_sqrt"),
        ("_mean", "_covariance"),
        ("_mean", "_scale_tril"),
        ("_mean", "_precision"),
        ("_info_vec", "prec_sqrt"),
        ("_info_vec", "_covariance"),
        ("_info_vec", "_scale_tril"),
        ("_info_vec", "_precision"),
    ],
)
def test_meta(loc, scale):
    names = "ijxyz"
    shapes = [Bint[2], Bint[3], Real, Reals[4], Reals[3, 2]]
    for real_inputs in ["x", "y", "z", "xy", "xz", "yz", "xyz"]:
        for int_inputs in ["", "i", "j", "ij"]:
            inputs = OrderedDict(
                (k, d) for k, d in zip(names, shapes) if k in int_inputs + real_inputs
            )
            expected = random_gaussian(inputs)

            kwargs = {
                scale.strip("_"): getattr(expected, scale),
                loc.strip("_"): getattr(expected, loc),
                "inputs": expected.inputs,
            }
            actual = Gaussian(**kwargs)
            assert_close(
                getattr(actual, loc), getattr(expected, loc), atol=1e-3, rtol=1e-3
            )
            assert_close(
                getattr(actual, scale), getattr(expected, scale), atol=1e-3, rtol=1e-3
            )
            assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "expr,expected_type",
    [
        ("-g1", Unary),
        ("g1 + 1", Contraction),
        ("g1 - 1", Contraction),
        ("1 + g1", Contraction),
        ("g1 + shift", Contraction),
        ("g1 + shift", Contraction),
        ("shift + g1", Contraction),
        ("shift - g1", Contraction),
        ("g1 + g1", (Gaussian, Contraction)),
        ("(g1 + g2 + g2) - g2", (Gaussian, Contraction)),
        ("g1(i=i0)", Gaussian),
        ("g2(i=i0)", Gaussian),
        ("g1(i=i0) + g2(i=i0)", Gaussian),
        ("g1(i=i0) + g2", Gaussian),
        ("g1(x=x0)", Tensor),
        ("g2(y=y0)", Tensor),
        ("(g1 + g2)(i=i0)", Gaussian),
        ("(g1 + g2)(x=x0, y=y0)", Tensor),
        ("(g2 + g1)(x=x0, y=y0)", Tensor),
        ('g1.reduce(ops.logaddexp, "x")', Tensor),
        ('(g1 + g2).reduce(ops.logaddexp, "x")', Contraction),
        ('(g1 + g2).reduce(ops.logaddexp, "y")', Contraction),
        ('(g1 + g2).reduce(ops.logaddexp, frozenset(["x", "y"]))', Tensor),
    ],
    ids=str,
)
def test_smoke(expr, expected_type):
    g1 = Gaussian(
        white_vec=numeric_array([[0.0, 0.1, 0.2], [2.0, 3.0, 4.0]]),
        prec_sqrt=ops.cholesky(
            numeric_array(
                [
                    [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]],
                    [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]],
                ]
            )
        ),
        inputs=OrderedDict([("i", Bint[2]), ("x", Reals[3])]),
    )
    assert isinstance(g1, Gaussian)

    g2 = Gaussian(
        white_vec=numeric_array([[0.0, 0.1], [2.0, 3.0]]),
        prec_sqrt=ops.cholesky(
            numeric_array([[[1.0, 0.2], [0.2, 1.0]], [[1.0, 0.2], [0.2, 1.0]]])
        ),
        inputs=OrderedDict([("i", Bint[2]), ("y", Reals[2])]),
    )
    assert isinstance(g2, Gaussian)

    shift = Tensor(numeric_array([-1.0, 1.0]), OrderedDict([("i", Bint[2])]))
    assert isinstance(shift, Tensor)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(numeric_array([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    y0 = Tensor(
        numeric_array([[0.2, 0.3], [0.8, 0.9]]), inputs=OrderedDict([("i", Bint[2])])
    )
    assert isinstance(y0, Tensor)

    result = eval(expr)
    assert isinstance(result, expected_type)

    print("Pretty:")
    pprint.pprint(result)


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "real_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("x", Reals[4])]),
        OrderedDict([("x", Reals[2, 3])]),
        OrderedDict([("x", Real), ("y", Real)]),
        OrderedDict([("x", Reals[2]), ("y", Reals[3])]),
        OrderedDict([("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]),
    ],
    ids=id_from_inputs,
)
def test_align(int_inputs, real_inputs):
    inputs1 = OrderedDict(
        list(sorted(int_inputs.items())) + list(sorted(real_inputs.items()))
    )
    inputs2 = OrderedDict(reversed(inputs1.items()))
    g1 = random_gaussian(inputs1)
    g2 = g1.align(tuple(inputs2))
    assert g2.inputs == inputs2
    g3 = g2.align(tuple(inputs1))
    assert_close(g3, g1)


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "real_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("x", Reals[4])]),
        OrderedDict([("x", Reals[2, 3])]),
        OrderedDict([("x", Real), ("y", Real)]),
        OrderedDict([("x", Reals[2]), ("y", Reals[3])]),
        OrderedDict([("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]),
    ],
    ids=id_from_inputs,
)
def test_eager_subs_mean(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)
    g = random_gaussian(inputs)

    # Check that Gaussian log density at its mean is zero.
    mean = g._mean
    means = {}
    start = 0
    for k, d in g.inputs.items():
        if d.dtype == "real":
            stop = start + d.num_elements
            data = mean[..., start:stop].reshape(mean.shape[:-1] + d.shape)
            means[k] = Tensor(data, int_inputs)
            start = stop
    actual = g(**means)
    expected_data = zeros(tuple(d.size for d in int_inputs.values()))
    expected = Tensor(expected_data, int_inputs)
    assert_close(actual, expected, atol=1e-5, rtol=None)


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "real_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("x", Reals[4])]),
        OrderedDict([("x", Reals[2, 3])]),
        OrderedDict([("x", Real), ("y", Real)]),
        OrderedDict([("x", Reals[2]), ("y", Reals[3])]),
        OrderedDict([("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]),
    ],
    ids=id_from_inputs,
)
def test_eager_subs(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)

    for order in itertools.permutations(inputs):
        ground_values = {}
        dependent_values = {}
        for i, name in enumerate(order):
            upstream = OrderedDict(
                [(k, inputs[k]) for k in order[:i] if k in int_inputs]
            )
            value = random_tensor(upstream, inputs[name])
            ground_values[name] = value(**ground_values)
            dependent_values[name] = value

        expected = g(**ground_values)
        actual = g
        for k in reversed(order):
            actual = actual(**{k: dependent_values[k]})
        assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_eager_subs_variable():
    inputs = OrderedDict([("i", Bint[2]), ("x", Real), ("y", Reals[2])])
    g1 = random_gaussian(inputs)

    g2 = g1(x="z")
    assert set(g2.inputs) == {"i", "y", "z"}
    assert g2.white_vec is g1.white_vec
    assert g2.prec_sqrt is g1.prec_sqrt

    g2 = g1(x="y", y="x")
    assert set(g2.inputs) == {"i", "x", "y"}
    assert g2.inputs["x"] == Reals[2]
    assert g2.white_vec is g1.white_vec
    assert g2.prec_sqrt is g1.prec_sqrt

    g2 = g1(i="j")
    assert set(g2.inputs) == {"j", "x", "y"}
    assert g2.white_vec is g1.white_vec
    assert g2.prec_sqrt is g1.prec_sqrt


@pytest.mark.parametrize(
    "subs",
    [
        (("x", 'Variable("u", Real) * 2'),),
        (("y", 'Variable("v", Reals[4]) + 1'),),
        (("z", 'Variable("w", Reals[6]).reshape((2,3))'),),
        (("x", 'Variable("v", Reals[4]).sum()'), ("y", 'Variable("v", Reals[4]) - 1')),
        (
            ("x", 'Variable("u", Real) * 2 + 1'),
            ("y", 'Variable("u", Real) * Tensor(ones((4,)))'),
            ("z", 'Variable("u", Real) * Tensor(ones((2, 3)))'),
        ),
        (
            (
                "y",
                'Einsum("abc,bc->a", Tensor(randn((4, 3, 5))), Variable("v", Reals[3, 5]))',
            ),
        ),
    ],
)
@pytest.mark.parametrize("g_ints", ["", "i", "j", "ij"])
@pytest.mark.parametrize("subs_ints", ["", "i", "j", "ji"])
def test_eager_subs_affine(subs, g_ints, subs_ints):
    sizes = {"i": 5, "j": 6}
    subs_inputs = OrderedDict((k, Bint[sizes[k]]) for k in subs_ints)
    g_inputs = OrderedDict((k, Bint[sizes[k]]) for k in g_ints)
    g_inputs["x"] = Real
    g_inputs["y"] = Reals[4]
    g_inputs["z"] = Reals[2, 3]
    g = random_gaussian(g_inputs)
    subs = {k: eval(v) + random_tensor(subs_inputs) for k, v in subs}

    inputs = g.inputs.copy()
    for v in subs.values():
        inputs.update(v.inputs)
    grounding_subs = {k: random_tensor(OrderedDict(), d) for k, d in inputs.items()}
    ground_subs = {k: v(**grounding_subs) for k, v in subs.items()}

    g_subs = g(**subs)
    assert issubclass(type(g_subs), (Gaussian, GaussianMixture))
    actual = g_subs(**grounding_subs)
    expected = g(**ground_subs)(**grounding_subs)
    assert_close(actual, expected, atol=1e-3, rtol=2e-4)


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "real_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("x", Reals[4])]),
        OrderedDict([("x", Reals[2, 3])]),
        OrderedDict([("x", Real), ("y", Real)]),
        OrderedDict([("x", Reals[2]), ("y", Reals[3])]),
        OrderedDict([("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]),
    ],
    ids=id_from_inputs,
)
def test_add_gaussian_number(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    n = Number(1.234)
    values = {
        name: random_tensor(int_inputs, domain) for name, domain in real_inputs.items()
    }

    assert_close((g + n)(**values), g(**values) + n, atol=1e-5, rtol=1e-5)
    assert_close((n + g)(**values), n + g(**values), atol=1e-5, rtol=1e-5)
    assert_close((g - n)(**values), g(**values) - n, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "real_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("x", Reals[4])]),
        OrderedDict([("x", Reals[2, 3])]),
        OrderedDict([("x", Real), ("y", Real)]),
        OrderedDict([("x", Reals[2]), ("y", Reals[3])]),
        OrderedDict([("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]),
    ],
    ids=id_from_inputs,
)
def test_add_gaussian_tensor(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    t = random_tensor(int_inputs, Real)
    values = {
        name: random_tensor(int_inputs, domain) for name, domain in real_inputs.items()
    }

    assert_close((g + t)(**values), g(**values) + t, atol=1e-5, rtol=1e-5)
    assert_close((t + g)(**values), t + g(**values), atol=1e-5, rtol=1e-5)
    assert_close((g - t)(**values), g(**values) - t, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "lhs_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("y", Reals[4])]),
        OrderedDict([("z", Reals[2, 3])]),
        OrderedDict([("x", Real), ("y", Reals[4])]),
        OrderedDict([("y", Reals[4]), ("z", Reals[2, 3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "rhs_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("y", Reals[4])]),
        OrderedDict([("z", Reals[2, 3])]),
        OrderedDict([("x", Real), ("y", Reals[4])]),
        OrderedDict([("y", Reals[4]), ("z", Reals[2, 3])]),
    ],
    ids=id_from_inputs,
)
def test_add_gaussian_gaussian(lhs_inputs, rhs_inputs):
    lhs_inputs = OrderedDict(sorted(lhs_inputs.items()))
    rhs_inputs = OrderedDict(sorted(rhs_inputs.items()))
    inputs = lhs_inputs.copy()
    inputs.update(rhs_inputs)
    int_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype != "real")
    real_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype == "real")

    g1 = random_gaussian(lhs_inputs)
    g2 = random_gaussian(rhs_inputs)
    values = {
        name: random_tensor(int_inputs, domain) for name, domain in real_inputs.items()
    }

    assert_close((g1 + g2)(**values), g1(**values) + g2(**values), atol=1e-4, rtol=None)


@pytest.mark.parametrize(
    "inputs",
    [
        OrderedDict([("i", Bint[2]), ("x", Real)]),
        OrderedDict([("i", Bint[3]), ("x", Real)]),
        OrderedDict([("i", Bint[2]), ("x", Reals[2])]),
        OrderedDict([("i", Bint[2]), ("x", Real), ("y", Real)]),
        OrderedDict([("i", Bint[3]), ("j", Bint[4]), ("x", Reals[2])]),
    ],
    ids=id_from_inputs,
)
def test_reduce_add(inputs):
    g = random_gaussian(inputs)
    actual = g.reduce(ops.add, "i")

    gs = [g(i=i) for i in range(g.inputs["i"].dtype)]
    expected = reduce(ops.add, gs)
    assert_close(actual, expected, rtol=None)


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "real_inputs",
    [
        OrderedDict([("x", Real), ("y", Real)]),
        OrderedDict([("x", Reals[2]), ("y", Reals[3])]),
        OrderedDict([("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]),
        OrderedDict(
            [("w", Reals[5]), ("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]
        ),
    ],
    ids=id_from_inputs,
)
def test_reduce_logsumexp(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    g_xy = g.reduce(ops.logaddexp, frozenset(["x", "y"]))
    assert_close(
        g_xy,
        g.reduce(ops.logaddexp, "x").reduce(ops.logaddexp, "y"),
        atol=1e-3,
        rtol=None,
    )
    assert_close(
        g_xy,
        g.reduce(ops.logaddexp, "y").reduce(ops.logaddexp, "x"),
        atol=1e-3,
        rtol=None,
    )


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
def test_reduce_logsumexp_partial(int_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(
        [("w", Reals[2]), ("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]
    )
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    batch_shape = tuple(d.size for d in int_inputs.values())
    all_values = {
        k: Tensor(randn(batch_shape + v.shape), int_inputs)
        for k, v in real_inputs.items()
    }
    real_vars = frozenset("wxyz")
    subsets = "w x y z wx wy wz xy xz yz wxy wxz wyz xyz".split()
    for reduced_vars in map(frozenset, subsets):
        values = {k: v for k, v in all_values.items() if k not in reduced_vars}

        # Check two ways of completely marginalizing.
        expected = g.reduce(ops.logaddexp, real_vars)
        actual = g.reduce(ops.logaddexp, reduced_vars).reduce(
            ops.logaddexp, real_vars - reduced_vars
        )
        assert_close(actual, expected, atol=1e-4, rtol=None)

        # Check two ways of substituting.
        expected = g(**values).reduce(ops.logaddexp, reduced_vars)
        actual = g.reduce(ops.logaddexp, reduced_vars)(**all_values)
        assert_close(actual, expected, atol=1e-4, rtol=None)


@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
def test_sample_partial(int_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(
        [("w", Reals[2]), ("x", Reals[4]), ("y", Reals[2, 3]), ("z", Real)]
    )
    inputs = int_inputs.copy()
    inputs.update(real_inputs)
    flat = ops.cat(
        [Variable(k, d).reshape((d.num_elements,)) for k, d in real_inputs.items()]
    )

    def compute_moments(samples):
        flat_samples = flat(**extract_samples(samples))
        assert set(flat_samples.inputs) == {"particle"} | set(int_inputs)
        mean = flat_samples.reduce(ops.mean)
        diff = flat_samples - mean
        cov = (diff[:, None] - diff[None, :]).reduce(ops.mean)
        return mean, cov

    sample_inputs = OrderedDict(particle=Bint[50000])
    rng_keys = [None] * 3
    if get_backend() == "jax":
        import jax.random

        rng_keys = jax.random.split(np.array([0, 0], dtype=np.uint32), 3)

    g = random_gaussian(inputs)
    all_vars = frozenset("wxyz")
    samples = g.sample(all_vars, sample_inputs, rng_keys[0])
    expected_mean, expected_cov = compute_moments(samples)
    subsets = "w x y z wx wy wz xy xz yz wxy wxz wyz xyz".split()
    for sampled_vars in map(frozenset, subsets):
        g2 = g.sample(sampled_vars, sample_inputs, rng_keys[1])
        samples = g2.sample(all_vars, sample_inputs, rng_keys[2])
        actual_mean, actual_cov = compute_moments(samples)
        assert_close(actual_mean, expected_mean, atol=1e-1, rtol=1e-1)
        assert_close(actual_cov, expected_cov, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("int_inputs", [{}, {"i": Bint[2]}], ids=id_from_inputs)
@pytest.mark.parametrize(
    "real_inputs",
    [{"x": Real}, {"x": Reals[4]}, {"x": Reals[2, 3]}],
    ids=id_from_inputs,
)
def test_integrate_variable(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    log_measure = random_gaussian(inputs)
    integrand = reduce(ops.add, [Variable(k, d) for k, d in real_inputs.items()])
    reduced_vars = frozenset(real_inputs)

    rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
    sampled_log_measure = log_measure.sample(
        reduced_vars, OrderedDict(particle=Bint[100000]), rng_key=rng_key
    )
    approx = Integrate(sampled_log_measure, integrand, reduced_vars)
    approx = approx.reduce(ops.mean, "particle")
    assert isinstance(approx, Tensor)

    exact = Integrate(log_measure, integrand, reduced_vars)
    assert isinstance(exact, Tensor)
    assert_close(approx, exact, atol=0.1, rtol=0.1)


@pytest.mark.xfail(get_backend() == "jax", reason="numerically unstable in jax backend")
@pytest.mark.parametrize(
    "int_inputs",
    [
        OrderedDict(),
        OrderedDict([("i", Bint[2])]),
        OrderedDict([("i", Bint[2]), ("j", Bint[3])]),
    ],
    ids=id_from_inputs,
)
@pytest.mark.parametrize(
    "real_inputs",
    [
        OrderedDict([("x", Real)]),
        OrderedDict([("x", Reals[2])]),
        OrderedDict([("x", Real), ("y", Real)]),
        OrderedDict([("x", Reals[2]), ("y", Reals[3])]),
        OrderedDict([("x", Reals[4]), ("y", Reals[2, 3])]),
    ],
    ids=id_from_inputs,
)
def test_integrate_gaussian(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    log_measure = random_gaussian(inputs)
    integrand = random_gaussian(inputs)
    reduced_vars = frozenset(real_inputs)

    rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
    sampled_log_measure = log_measure.sample(
        reduced_vars, OrderedDict(particle=Bint[100000]), rng_key=rng_key
    )
    approx = Integrate(sampled_log_measure, integrand, reduced_vars)
    approx = approx.reduce(ops.mean, "particle")
    assert isinstance(approx, Tensor)

    exact = Integrate(log_measure, integrand, reduced_vars)
    assert isinstance(exact, Tensor)
    assert_close(approx, exact, atol=0.1, rtol=0.1)


def test_mc_plate_gaussian():
    log_measure = Gaussian(
        white_vec=numeric_array([0.0]),
        prec_sqrt=numeric_array([[1.0]]),
        inputs=(("loc", Real),),
    ) + numeric_array(-0.9189)

    plate_size = 10
    integrand = Gaussian(
        white_vec=randn((plate_size, 1)) + 3.0,
        prec_sqrt=ones((plate_size, 1, 1)),
        inputs=(("data", Bint[plate_size]), ("loc", Real)),
    )

    rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
    res = Integrate(log_measure.sample("loc", rng_key=rng_key), integrand, "loc")
    res = res.reduce(ops.mul, "data")
    assert not ((res == float("inf")) | (res == float("-inf"))).any()


def test_eager_add():
    g1 = Gaussian(randn((2,)), randn((1, 2)), OrderedDict(a=Real))
    g2 = Gaussian(randn((1,)), randn((1, 1)), OrderedDict(a=Real))
    a = Variable("a", Real)

    actual = (g1 + g2).reduce(ops.logaddexp)
    assert isinstance(actual, Tensor)

    actual = Contraction(ops.logaddexp, ops.add, frozenset({a}), (g1, g2))
    assert isinstance(actual, Tensor)


@pytest.mark.parametrize("interp", [eager, lazy])
def test_nested_subs(interp):
    with interp:
        g = Gaussian(randn(3), randn(2, 3), OrderedDict([("b", Real), ("a", Real)]))
        a = ops.abs(Variable("aux_0", Real))
        b = ops.abs(Variable("aux_1", Real))
        g_ab = g(a=a, b=b)
        g_a_b = g(a=a)(b=b)
        g_b_a = g(b=b)(a=a)

    # Test subs fusion.
    assert isinstance(g_ab, Subs)
    assert isinstance(g_ab.arg, Gaussian)
    assert isinstance(g_a_b, Subs)
    assert isinstance(g_a_b.arg, Gaussian)
    assert isinstance(g_b_a, Subs)
    assert isinstance(g_b_a.arg, Gaussian)

    # Compare on ground data.
    subs = {"aux_0": randn(()), "aux_1": randn(())}
    assert_close(g_ab(**subs), g_a_b(**subs), atol=1e-3, rtol=1e-3)
    assert_close(g_ab(**subs), g_b_a(**subs), atol=1e-3, rtol=1e-3)
