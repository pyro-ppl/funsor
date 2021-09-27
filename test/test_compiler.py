# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import pytest

import funsor.ops as ops
from funsor.compiler import compile_funsor
from funsor.domains import Real, Reals
from funsor.interpretations import reflect
from funsor.optimizer import apply_optimizer
from funsor.sum_product import sum_product
from funsor.tensor import Tensor
from funsor.terms import Number, Tuple, Variable
from funsor.testing import assert_close, randn


@functools.singledispatch
def extract_data(x):
    raise TypeError(type(x).__name__)


@extract_data.register(Number)
@extract_data.register(Tensor)
def _(x):
    return x.data


@extract_data.register(Tuple)
def _(x):
    return tuple(extract_data(arg) for arg in x.args)


def check_compiler(expr):
    # Create a random substitution.
    subs = {k: randn(d.shape) for k, d in expr.inputs.items()}
    expected = expr(**subs)
    expected_data = extract_data(expected)

    # Execute a funsor program.
    program = compile_funsor(expr)
    actual = program(**subs)
    assert_close(actual, expected_data)

    # Execute a printed program.
    code = program.as_code(name="program2")
    print(code)
    env = {}
    exec(code, None, env)
    actual = env["program2"](**subs)
    assert_close(actual, expected_data)


def test_lowered_1():
    x = Variable("x", Reals[3])
    check_compiler(x)


def test_lowered_2():
    x = Variable("x", Reals[3])
    y = x * x
    check_compiler(y)


def test_lowered_3():
    x = Variable("x", Reals[3])
    y = 1 + x * x
    z = y[0] * y[1] + y[2]
    check_compiler(z)


def test_lowered_4():
    x = Variable("x", Real)
    y = Variable("y", Real)
    z = Tuple((Number(1), x, y, x * y))
    check_compiler(z)


@pytest.mark.xfail(reason="Bound variable lowering is not yet supported")
def test_sum_product():
    factors = [
        Variable("f", Reals[5])["x"],
        Variable("g", Reals[5, 4])["x", "y"],
        Variable("h", Reals[4, 3, 2])["y", "z", "i"],
    ]
    eliminate = frozenset({"x", "y", "z", "i"})
    plates = frozenset({"i"})
    with reflect:
        expr = sum_product(ops.logaddexp, ops.add, factors, eliminate, plates)
        expr = apply_optimizer(expr)
    assert set(expr.inputs) == {"f", "g", "h"}

    check_compiler(expr)
