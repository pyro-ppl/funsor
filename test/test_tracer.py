# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor.ops as ops
from funsor.domains import Real
from funsor.interpretations import reflect
from funsor.op_factory import make_op
from funsor.optimizer import apply_optimizer
from funsor.sum_product import sum_product
from funsor.tensor import Tensor
from funsor.testing import assert_close, randn
from funsor.tracer import trace_function


def test_sum_product():
    def fn(f, g, h):
        # This function only uses Funsors internally.
        factors = [
            Tensor(f)["x"],
            Tensor(g)["x", "y"],
            Tensor(h)["y", "z", "i"],
        ]
        eliminate = frozenset({"x", "y", "z", "i"})
        plates = frozenset({"i"})
        with reflect:
            expr = sum_product(ops.logaddexp, ops.add, factors, eliminate, plates)
        expr = apply_optimizer(expr)
        return expr.data

    f = randn(5)
    g = randn(5, 4)
    h = randn(4, 3, 2)
    expected = fn(f, g, h)

    traced_fn = trace_function(fn, dict(f=f, g=g, h=h))
    actual = traced_fn(f=f, g=g, h=h)
    assert_close(actual, expected)


def test_wrapped_op():
    @make_op
    def wrapper(x: Real) -> Real:
        # This should not be traced.
        return ops.add(x, dict(x=x))

    def fn(x):
        y = ops.add(x, 1)
        z = wrapper(y)
        return ops.add(z, 1)

    x = randn(())
    expected = fn(x)

    traced_fn = trace_function(fn, x)
    actual = traced_fn(x=x)
    assert_close(actual, expected)
