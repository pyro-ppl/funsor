# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict
from functools import reduce

import pytest
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


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_mul_x_y(sum_op, prod_op, tojvp):
    with autodiff:
        # Mul
        x = tojvp(random_tensor(OrderedDict(j=Bint[4])))
        y = tojvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = prod_op(x, y)
        result = grad(z, (x, y), out_adj)

    expected_x = prod_op(out_adj, y.primal).reduce(sum_op, "k")
    expected_y = prod_op(out_adj, x.primal)

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x, rtol=1e-5)
    assert_close(actual_y, expected_y, rtol=1e-5)


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_mul_x_x(sum_op, prod_op, tojvp):
    with autodiff:
        # Mul
        x = tojvp(random_tensor(OrderedDict(j=Bint[4])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = prod_op(x, x)
        result = grad(z, (x,), out_adj)

    two = 2 if tojvp is to_jvp else math.log(2)
    expected_x = reduce(prod_op, (two, out_adj, x.primal))
    actual_x = apply_optimizer(result[x])
    assert_close(actual_x, expected_x)


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_add_x_x(sum_op, prod_op, tojvp):
    with autodiff:
        # Add
        x = tojvp(random_tensor(OrderedDict(j=Bint[4])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = sum_op(x, x)
        result = grad(z, (x,), out_adj)

    two = 2 if tojvp is to_jvp else math.log(2)
    expected_x = prod_op(two, out_adj)
    actual_x = apply_optimizer(result[x])
    assert_close(actual_x, expected_x)


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_add_x_y(sum_op, prod_op, tojvp):
    with autodiff:
        # Add
        x = tojvp(random_tensor(OrderedDict(j=Bint[4])))
        y = tojvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = sum_op(x, y)
        result = grad(z, (x, y), out_adj)

    expected_x = out_adj.reduce(sum_op, "k")
    expected_y = out_adj

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_mul_add_x_x_y(sum_op, prod_op, tojvp):
    with autodiff:
        # Add Mul
        x = tojvp(random_tensor(OrderedDict(j=Bint[4])))
        y = tojvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = sum_op(prod_op(x, x), y)
        result = grad(z, (x, y), out_adj)

    two = 2 if tojvp is to_jvp else math.log(2)
    expected_x = reduce(prod_op, (two, x.primal, out_adj.reduce(sum_op, "k")))
    expected_y = out_adj

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_mul_add_xx_yy(sum_op, prod_op, tojvp):
    with autodiff:
        # Add Mul
        x = tojvp(random_tensor(OrderedDict(j=Bint[4])))
        y = tojvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))

        z = reduce(sum_op, (prod_op(x, x), y, y))
        result = grad(z, (x, y), out_adj)

    two = 2 if tojvp is to_jvp else math.log(2)
    expected_x = reduce(prod_op, (two, x.primal, out_adj.reduce(sum_op, "k")))
    expected_y = prod_op(two, out_adj)

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x)
    assert_close(actual_y, expected_y)


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_reduce_add_x(sum_op, prod_op, tojvp):
    with autodiff:
        # Reduce
        y = tojvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = y.reduce(sum_op, "k")
        result = grad(z, (y,), out_adj)

    expected_y = out_adj.expand((Variable("k", Bint[5]),))
    actual_y = apply_optimizer(result[y])
    assert_close(actual_y, expected_y, rtol=1e-5)


@pytest.mark.parametrize(
    "sum_op,prod_op,div_op,tojvp",
    [
        (ops.add, ops.mul, ops.safediv, to_jvp),
        (ops.logaddexp, ops.add, ops.safesub, to_ljvp),
    ],
)
def test_reduce_mul_x(sum_op, prod_op, div_op, tojvp):
    with autodiff:
        # Reduce
        y = tojvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(j=Bint[4]))

        z = y.reduce(prod_op, "k")
        result = grad(z, (y,), out_adj)

    actual_y = apply_optimizer(result[y])
    expected_y = div_op(prod_op(out_adj, z.primal), y.primal)
    assert_close(actual_y, apply_optimizer(expected_y), rtol=1e-5)


@pytest.mark.parametrize(
    "sum_op,prod_op,tojvp",
    [(ops.add, ops.mul, to_jvp), (ops.logaddexp, ops.add, to_ljvp)],
)
def test_mul_reduce_x_y(sum_op, prod_op, tojvp):
    with autodiff:
        # Reduce
        x = tojvp(random_tensor(OrderedDict(j=Bint[4])))
        y = tojvp(random_tensor(OrderedDict(j=Bint[4], k=Bint[5])))
        out_adj = random_tensor(OrderedDict(k=Bint[5]))

        z = prod_op(x, y).reduce(sum_op, "j")
        result = grad(z, (x, y), out_adj)

    expected_x = prod_op(y.primal, out_adj).reduce(sum_op, "k")
    expected_y = prod_op(x.primal, out_adj.expand((Variable("j", Bint[4]),)))

    actual_x = apply_optimizer(result[x])
    actual_y = apply_optimizer(result[y])

    assert_close(actual_x, expected_x, rtol=1e-5)
    assert_close(actual_y, expected_y, rtol=1e-5)


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
