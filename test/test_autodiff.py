# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import torch

import funsor
import funsor.ops as ops
from funsor.autodiff import JVP, grad, to_jvp, to_ljvp
from funsor.domains import Bint, Real, Reals
from funsor.factory import Bound, Fresh, Has, make_funsor
from funsor.interpretations import autodiff, trace
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer
from funsor.sum_product import MarkovProduct
from funsor.tensor import Tensor
from funsor.terms import Binary, Funsor, Lambda, Number, Tuple, Variable, lazy
from funsor.testing import assert_close, random_tensor

funsor.set_backend("torch")


def test_mul_x_y():
    with autodiff:
        # Mul
        x = to_jvp(random_tensor(OrderedDict(j=Bint[4])))
        y = to_jvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
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
        x = to_jvp(random_tensor(OrderedDict(j=Bint[4])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = x * x
        result = grad(z, (x,), out_adj)

    expected_x = 2 * out_adj * x[0]
    actual_x = apply_optimizer(result[x])
    assert_close(actual_x, expected_x)


def test_add_x_x():
    with autodiff:
        # Add
        x = to_jvp(random_tensor(OrderedDict(j=Bint[4])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = x + x
        result = grad(z, (x,), out_adj)

    expected_x = 2 * out_adj
    actual_x = apply_optimizer(result[x])
    assert_close(actual_x, expected_x)


def test_add_x_y():
    with autodiff:
        # Add
        x = to_jvp(random_tensor(OrderedDict(j=Bint[4])))
        y = to_jvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
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
        x = to_jvp(random_tensor(OrderedDict(j=Bint[4])))
        y = to_jvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
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
        x = to_jvp(random_tensor(OrderedDict(j=Bint[4])))
        y = to_jvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
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
        y = to_jvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = y.reduce(ops.add, "k")
        result = grad(z, (y,), out_adj)

    expected_y = out_adj.expand(ops.add, (Variable("k", Bint[5]),))
    actual_y = apply_optimizer(result[y])
    assert_close(actual_y, expected_y)


def test_mul_reduce_x_y():
    with autodiff:
        # Reduce
        x = to_jvp(random_tensor(OrderedDict(j=Bint[4])))
        y = to_jvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(k=Bint[5]))

        z = (x * y).reduce(ops.add, "j")
        result = grad(z, (x, y), out_adj)

    expected_x = (y[0] * out_adj).reduce(ops.add, "k")
    expected_y = x[0] * out_adj.expand(ops.add, (Variable("j", Bint[4]),))

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


#  def test_trace():
#      @make_funsor
#      def Matmul(
#          x: Has[{"i"}],
#          y: Has[{"i"}],
#          i: Bound
#      ) -> Fresh[lambda x: x]:
#          return (x * y).reduce(ops.add, i)
#
#      x = random_tensor(OrderedDict(j=Bint[4]))
#      y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
#      eager_z = Matmul(x, y, "j")
#      with lazy:
#          lazy_z = Matmul(x, y, "j")
#
#      with trace:
#          trace_z = Matmul(x, y, "j")
#
#      assert_close(eager_z, apply_optimizer(lazy_z))
#      assert_close(eager_z, apply_optimizer(trace_z))
