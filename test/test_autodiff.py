# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.autodiff import JVP, to_var, to_arg, fjit, grad, requires_grad, transpose
from funsor.testing import assert_close, random_tensor
from funsor.terms import Variable, Number, lazy, Lambda, Binary, Funsor, Tuple
from funsor.tensor import Tensor
from funsor.optimizer import apply_optimizer
from funsor.interpreter import interpretation
from funsor.factory import make_funsor, Bound, Fresh, Has
from funsor.sum_product import MarkovProduct
from funsor.interpretations import trace, autodiff


import torch
import funsor
funsor.set_backend("torch")


def test_mul_x_y():
    with autodiff:
        # Mul
        x = requires_grad(random_tensor(OrderedDict(j=Bint[4])))
        y = requires_grad(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = x * y
        result = grad(z, (x, y), out_adj)

    expected_x = (out_adj * y[0]).reduce(ops.add, "k")
    expected_y = out_adj * x[0]

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


def test_mul_x_x():
    with autodiff:
        # Mul
        x = requires_grad(random_tensor(OrderedDict(j=Bint[4])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = x * x
        result = grad(z, (x,), out_adj)

    expected_x = 2 * out_adj * x[0]
    actual_x = apply_optimizer(result[x])
    assert_close(actual_x, expected_x)


def  test_add_x_x():
    with autodiff:
        # Add
        x = requires_grad(random_tensor(OrderedDict(j=Bint[4])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = x + x
        result = grad(z, (x,), out_adj)

    expected_x = 2 * out_adj
    actual_x = apply_optimizer(result[x])
    assert_close(actual_x, expected_x)


def test_add_x_y():
    with autodiff:
        # Add
        x = requires_grad(random_tensor(OrderedDict(j=Bint[4])))
        y = requires_grad(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = x + y
        result = grad(z, (x, y), out_adj)

    expected_x = out_adj.reduce(ops.add, "k")
    expected_y = out_adj

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


def test_mul_add_x_x_y():
    with autodiff:
        # Add Mul
        x = requires_grad(random_tensor(OrderedDict(j=Bint[4])))
        y = requires_grad(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = x * x + y
        result = grad(z, (x, y), out_adj)

    expected_x = 2 * x[0] * out_adj.reduce(ops.add, "k")
    expected_y = out_adj

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


def test_mul_add_xx_yy():
    with autodiff:
        # Add Mul
        x = requires_grad(random_tensor(OrderedDict(j=Bint[4])))
        y = requires_grad(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = x * x + y + y
        result = grad(z, (x, y), out_adj)

    expected_x = 2 * x[0] * out_adj.reduce(ops.add, "k")
    expected_y = 2 * out_adj

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


def test_reduce_x():
    with autodiff:
        # Reduce
        y = requires_grad(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = y.reduce(ops.add, "k")
        result = grad(z, (y,), out_adj)

    expected_y = out_adj.expand(ops.add, (Variable("k", Bint[5]),)).align(tuple(y[0].inputs.keys()))
    actual_y = apply_optimizer(result[y])
    assert_close(actual_y, expected_y)


def test_trace():
    @make_funsor
    def Matmul(
        x: Has[{"i"}],
        y: Has[{"i"}],
        i: Bound
    ) -> Fresh[lambda x: x]:
        return (x * y).reduce(ops.add, i)

    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    eager_z = Matmul(x, y, "j")
    with lazy:
        lazy_z = Matmul(x, y, "j")

    with trace:
        trace_z = Matmul(x, y, "j")

    assert_close(eager_z, apply_optimizer(lazy_z))
    assert_close(eager_z, apply_optimizer(trace_z))
