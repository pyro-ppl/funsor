# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.domains import Real, Reals
from funsor.gaussian import Gaussian
from funsor.interpretations import reflect
from funsor.interpreter import reinterpret
from funsor.op_factory import make_op
from funsor.ops.tracer import trace_function
from funsor.optimizer import apply_optimizer
from funsor.sum_product import sum_product
from funsor.tensor import Tensor
from funsor.terms import to_data, to_funsor
from funsor.testing import assert_close, randn


def check_tracer(fn, data):
    expected = fn(**data)

    traced_fn = trace_function(fn, data)
    actual = traced_fn(**data)
    assert_close(actual, expected)


def test_id():
    def fn(x):
        return x

    data = dict(x=randn(2, 3))
    check_tracer(fn, data)


def test_chain():
    def fn(x):
        for i in range(10):
            x = ops.mul(x, x)
        return x

    data = dict(x=randn(4))
    check_tracer(fn, data)


def test_wrapped_op():
    @make_op
    def wrapper(x: Real) -> Real:
        # This should not be traced.
        return ops.add(x, x)

    def fn(x):
        y = ops.add(x, 1)
        z = wrapper(y)
        return ops.add(z, 1)

    data = dict(x=randn(()))
    check_tracer(fn, data)


def test_use_funsor_interally_1():
    def fn(x, y, z):
        # Convert backend arrays -> to funsors.
        x = to_funsor(x)
        y = to_funsor(y)
        z = to_funsor(z)

        # Operate on funsors.
        result = x @ y + z + 1

        # Convert funsors -> to backend array.
        return to_data(result)

    data = dict(x=randn(2, 3), y=randn(3, 4), z=randn(2, 4))
    check_tracer(fn, data)


@pytest.mark.xfail(reason="TODO Gaussian directly uses backend, bypassing ops")
def test_use_funsor_interally_2():
    def gaussian_log_prob(white_vec, prec_sqrt, value):
        # Convert backend arrays -> to funsors.
        g = Gaussian(white_vec, prec_sqrt, OrderedDict(x=Reals[3]))
        value = to_funsor(value)

        # Operate on funsors.
        log_prob = g(x=value) - g.reduce(ops.logaddexp)

        # Convert funsors -> to backend array.
        return to_data(log_prob)

    prec_sqrt = randn(3, 3)
    data = dict(white_vec=randn(3), prec_sqrt=prec_sqrt, value=randn(3))
    check_tracer(gaussian_log_prob, data)


@pytest.mark.xfail(reason="TODO support tuples for multiple outputs")
def test_tuple():
    def fn(x, y):
        return (1, x, y, ops.mul(x, y))

    data = dict(x=randn(3), y=randn(2, 1))
    check_tracer(fn, data)


@pytest.mark.xfail(
    reason="funsor.cnf._eager_contract_tensors directly calls opt_einsum, bypassing ops"
)
def test_sum_product():
    def fn(f, g, h):
        # This function only uses Funsors internally.
        factors = [Tensor(f)["x"], Tensor(g)["x", "y"], Tensor(h)["y", "z", "i"]]
        eliminate = frozenset({"x", "y", "z", "i"})
        plates = frozenset({"i"})
        with reflect:
            expr = sum_product(ops.logaddexp, ops.add, factors, eliminate, plates)
            expr = apply_optimizer(expr)
        expr = reinterpret(expr)
        return expr.data

    data = dict(f=randn(5), g=randn(5, 4), h=randn(4, 3, 2))
    expected = fn(**data)

    traced_fn = trace_function(fn, data)
    actual = traced_fn(**data)
    assert_close(actual, expected)
