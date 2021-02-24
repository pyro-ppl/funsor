# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Bint
from funsor.jvp import Tangent
from funsor.testing import assert_close, random_tensor


def test_identity():
    x = random_tensor(OrderedDict(i=Bint[2]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    x_ = Tangent((x, dx))
    f, df = x_
    assert_close(f, x)
    assert_close(df, dx)


def test_log():
    x = random_tensor(OrderedDict(i=Bint[2]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    x_ = Tangent((x, dx))
    f, df = x_.log()
    assert_close(f, x.log())
    assert_close(df, dx / x)


def test_add():
    x = random_tensor(OrderedDict(i=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[3]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    dy = random_tensor(OrderedDict(j=Bint[3]))
    # dx = Number(1.0)  # Variable("dx", Real)
    # dy = Number(1.0)  # Variable("dy", Real)
    x_ = Tangent((x, dx))
    y_ = Tangent((y, dy))
    f_ = x_ + y_
    f, df = f_
    assert_close(f, x + y)
    assert_close(df, dx + dy)


def test_mul():
    x = random_tensor(OrderedDict(i=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[3]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    dy = random_tensor(OrderedDict(j=Bint[3]))
    # dx = Number(1.0)  # Variable("dx", Real)
    # dy = Number(0.0)  # Variable("dy", Real)
    x_ = Tangent((x, dx))
    y_ = Tangent((y, dy))
    f, df = x_ * y_
    assert_close(f, x * y)
    assert_close(df, (x * dy + y * dx).align(tuple(df.inputs.keys())))


def test_reduce_sum():
    x = random_tensor(OrderedDict(j=Bint[4]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    x_ = Tangent((x, dx))
    f, df = x_.reduce(ops.add, "j")
    assert_close(f, x.reduce(ops.add, "j"))
    assert_close(df, dx.reduce(ops.add, "j"))


def test_reduce_prod():
    x = random_tensor(OrderedDict(j=Bint[4]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    x_ = Tangent((x, dx))
    f, df = x_.reduce(ops.mul, "j")
    assert_close(f, x.reduce(ops.mul, "j"))
    assert_close(df, (f * dx / x).reduce(ops.add, "j"))


def test_matmul_tensor():
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    x_ = Tangent((x, dx))
    y_ = Tangent((y, dy))
    xy_ = x_ * y_
    z, dz = xy_.reduce(ops.add, "j")
    assert_close(z, (x * y).reduce(ops.add, "j"))
    assert_close(dz, (y * dx + x * dy).reduce(ops.add, "j"))


def test_compose():
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    x_ = Tangent((x, dx))
    logx_ = x_.log()
    y_ = Tangent((y, dy))
    logxy_ = logx_ * y_
    z, dz = logxy_.reduce(ops.add, "j")
    assert_close(z, (x.log() * y).reduce(ops.add, "j"))
    assert_close(dz, (y * dx / x + x.log() * dy).reduce(ops.add, "j"))
