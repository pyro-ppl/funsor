# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import reduce

import numpy as np
import pytest

from funsor import ops
from funsor.adjoint import adjoint
from funsor.approximations import (
    argmax_approximate,
    laplace_approximate,
    mean_approximate,
)
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import Bint, Real, Reals
from funsor.einsum import einsum
from funsor.interpretations import eager, normalize, reflect
from funsor.interpreter import reinterpret
from funsor.montecarlo import MonteCarlo
from funsor.tensor import Tensor
from funsor.terms import Approximate, Variable
from funsor.testing import (
    assert_close,
    make_einsum_example,
    random_gaussian,
    random_tensor,
    xfail_if_not_implemented,
    xfail_param,
)
from funsor.util import get_backend

monte_carlo = MonteCarlo(rng_key=np.array([0, 0], dtype=np.uint32))
monte_carlo_10 = MonteCarlo(
    rng_key=np.array([0, 0], dtype=np.uint32),
    particle=Bint[10],
)
particles_10 = frozenset([Variable("particle", Bint[10])])


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        monte_carlo,
        monte_carlo_10,
    ],
)
def test_tensor_smoke(approximate):
    with normalize:
        model = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
        guide = random_tensor(OrderedDict(j=Bint[3]))
        p = model.approximate(ops.logaddexp, guide, "j")
    with approximate, xfail_if_not_implemented():
        q = reinterpret(p)
    assert q.output == p.output
    if approximate == monte_carlo_10:
        assert q.input_vars.issubset(p.input_vars | particles_10)
    else:
        assert q.input_vars.issubset(p.input_vars)


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        laplace_approximate,
        xfail_param(mean_approximate, reason="alpha conversion bug"),
        monte_carlo,
        monte_carlo_10,
    ],
)
def test_gaussian_smoke(approximate):
    with normalize:
        model = random_gaussian(OrderedDict(i=Bint[2], u=Real, v=Reals[3]))
        guide = random_gaussian(OrderedDict(u=Real, v=Reals[3]))
        p = model.approximate(ops.logaddexp, guide, {"u", "v"})
    with approximate, xfail_if_not_implemented():
        q = reinterpret(p)
    assert q.output == p.output
    if approximate == monte_carlo_10:
        assert q.input_vars.issubset(p.input_vars | particles_10)
    else:
        assert q.input_vars.issubset(p.input_vars)


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        monte_carlo,
    ],
    ids=str,
)
def test_tensor_linear(approximate):
    m1 = random_tensor(OrderedDict(i=Bint[2], x=Bint[4]))
    m2 = random_tensor(OrderedDict(j=Bint[3], x=Bint[4]))
    s = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    guide = random_tensor(OrderedDict(i=Bint[2], j=Bint[3], x=Bint[4]))
    with approximate, xfail_if_not_implemented():
        expected = (m1 + s * m2).approximate(ops.logaddexp, guide, "x")
        q1 = m1.approximate(ops.logaddexp, guide, "x")
        q2 = m2.approximate(ops.logaddexp, guide, "x")
    actual = q1 + s * q2

    if approximate == monte_carlo:
        assert actual.inputs == expected.inputs
    else:
        assert_close(actual, expected)


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        laplace_approximate,
        mean_approximate,
        monte_carlo,
    ],
    ids=str,
)
def test_gaussian_linear(approximate):
    m1 = random_gaussian(OrderedDict(i=Bint[2], x=Real))
    m2 = random_gaussian(OrderedDict(j=Bint[3], x=Real))
    s = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    guide = random_gaussian(OrderedDict(i=Bint[2], j=Bint[3], x=Real))
    with approximate, xfail_if_not_implemented():
        expected = (m1 + s * m2).approximate(ops.logaddexp, guide, "x")
        q1 = m1.approximate(ops.logaddexp, guide, "x")
        q2 = m2.approximate(ops.logaddexp, guide, "x")
    actual = q1 + s * q2

    if approximate == monte_carlo:
        assert actual.inputs == expected.inputs
    else:
        assert_close(actual, expected)


def test_backward_argmax_simple_reduce():

    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))

    with reflect:
        y = x.reduce(ops.logaddexp, "i")

    with argmax_approximate:
        actual = adjoint(ops.logaddexp, ops.add, y)[x]
        actual = actual + x  # TODO do we expect this +x or not?

    assert actual.inputs == x.inputs
    assert actual.output == x.output

    expected_point = Tensor(ops.argmax(x.data, -2), dtype=2)["j"]
    expected_value = x.reduce(ops.max, "i")
    expected = expected_value + Delta("i", expected_point)

    assert_close(actual, expected)


def test_backward_argmax_simple_binary():

    x1 = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    x2 = random_tensor(OrderedDict(j=Bint[3], k=Bint[4]))
    approx_vars = x1.input_vars | x2.input_vars

    with reflect:
        y = (x1 + x2).reduce(ops.logaddexp, approx_vars)

    with argmax_approximate:
        expected = Approximate(ops.logaddexp, x1 + x2, x1 + x2, approx_vars)
        actuals = adjoint(ops.logaddexp, ops.add, y)

    actual = reduce(ops.add, [actuals[x] for x in (x1, x2)])
    assert actual.input_vars == expected.input_vars
    assert actual.output == expected.output

    assert isinstance(expected, Contraction)
    assert not expected.reduced_vars
    points, tensor = expected.terms
    assert isinstance(points, Delta)
    expected = points.align(tuple(actual.inputs)) + tensor

    assert_close(actual, expected)


def test_backward_argmax_simple_contraction():

    x1 = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    x2 = random_tensor(OrderedDict(j=Bint[3], k=Bint[4]))
    approx_vars = x1.input_vars | x2.input_vars

    with reflect:
        y = Contraction(ops.logaddexp, ops.add, approx_vars, (x1, x2))

    with argmax_approximate:
        expected = Approximate(ops.logaddexp, x1 + x2, x1 + x2, approx_vars)
        actuals = adjoint(ops.logaddexp, ops.add, y)

    actual = reduce(ops.add, [actuals[x] for x in (x1, x2)])
    assert actual.input_vars == expected.input_vars
    assert actual.output == expected.output

    assert isinstance(expected, Contraction)
    assert not expected.reduced_vars
    points, tensor = expected.terms
    assert isinstance(points, Delta)
    expected = points.align(tuple(actual.inputs)) + tensor

    assert_close(actual, expected)


@pytest.mark.parametrize(
    "equation,plates",
    [
        (",->", ""),
        ("a,b->", ""),
        ("ab,a->", ""),
        ("a,b,c->", ""),
        ("a,a->", ""),
        ("a,a,a,ab->", ""),
        ("abc,bcd,cde->", ""),
        ("abc,cbd,edc->", ""),
        ("ab,bc,cd->", ""),
        ("ab,b,bc,c,cd,d->", ""),
    ],
)
def test_einsum_adjoint_argmax(equation, plates):
    if get_backend() == "torch":
        backend = "pyro.ops.einsum.torch_map"
    elif get_backend() == "jax" or get_backend() == "numpy":
        backend = "funsor.einsum.numpy_map"

    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with reflect:
        fwd_expr = einsum(equation, *funsor_operands, plates=plates, backend=backend)

    expected = einsum(equation, *funsor_operands, plates=plates, backend=backend)

    with argmax_approximate:
        actuals = adjoint(ops.max, ops.add, fwd_expr)

    for fv in funsor_operands:
        actual = (actuals[fv] + fv).reduce(ops.max, fv.input_vars)
        assert_close(actual, expected)
