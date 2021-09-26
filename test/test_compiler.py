# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor.ops as ops
from funsor.compiler import FunsorProgram
from funsor.domains import Reals
from funsor.interpretations import reflect
from funsor.optimizer import apply_optimizer
from funsor.sum_product import sum_product
from funsor.tensor import Tensor
from funsor.terms import Number, Variable
from funsor.testing import assert_close, randn


def check_compiler(expr):
    # Create a random substitution.
    subs = {k: randn(d.shape) for k, d in expr.inputs.items()}

    # Create and execute a funsor program.
    program = FunsorProgram(expr)
    actual = program(**subs)

    # Compare with funsor substitution.
    expected = expr(**subs)
    assert isinstance(expected, (Number, Tensor))
    assert_close(actual, expected.data)


def test_simple_1():
    x = Variable("x", Reals[3])
    check_compiler(x)


def test_simple_2():
    x = Variable("x", Reals[3])
    y = x * x
    check_compiler(y)


def test_simple_3():
    x = Variable("x", Reals[3])
    y = 1 + x * x
    z = y[0] * y[1] + y[2]
    check_compiler(z)


def test_sum_product():
    factors = [
        Variable("f", Reals[5])["x"],
        Variable("g", Reals[5, 4])["x", "y"],
        Variable("h", Reals[4, 3, 2])["x", "y", "i"],
    ]
    eliminate = frozenset({"x", "y", "z", "i"})
    plates = frozenset({"i"})
    with reflect:
        expr = sum_product(ops.logaddexp, ops.add, factors, eliminate, plates)
        expr = apply_optimizer(expr)

    check_compiler(expr)
