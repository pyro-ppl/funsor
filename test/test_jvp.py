# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Bint, Real
from funsor.jvp import Tangent, JVP
from funsor.vjp import transpose
from funsor.testing import assert_close, random_tensor
from funsor.terms import Variable, Number, lazy
from funsor.tensor import Tensor
from funsor.optimizer import apply_optimizer
from funsor.interpreter import interpretation


import torch
import funsor
funsor.set_backend("torch")


def test_id():
    x = random_tensor(OrderedDict(i=Bint[2]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    x_ = JVP(x)
    with lazy:
        f = x_
    assert_close(f.primal, x)
    df = f.tangent[str(id(x))](**{str(id(x)): dx})
    assert_close(df, dx)


def test_log():
    x = Tensor(torch.tensor([1., 2.]), OrderedDict(i=Bint[2]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    x_ = JVP(x)
    with lazy:
        f = x_.log()
    primal = apply_optimizer(f.primal)
    assert_close(primal, x.log())
    df = f.tangent[str(id(x))](**{str(id(x)): dx})
    assert_close(df, dx / x)


def test_add():
    x = random_tensor(OrderedDict(i=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[3]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    dy = random_tensor(OrderedDict(j=Bint[3]))
    x_ = JVP(x)
    y_ = JVP(y)
    with lazy:
        f = x_ + y_

    primal = apply_optimizer(f.primal)
    assert_close(primal, x + y)

    dfdx = f.tangent[str(id(x))](**{str(id(x)): dx})
    assert_close(dfdx, dx+y-y)

    dfdy = f.tangent[str(id(y))](**{str(id(y)): dy})
    assert_close(dfdy, dy+x-x)


def test_add_two():
    x = random_tensor(OrderedDict(i=Bint[2]))
    dx = Tensor(torch.tensor([1, 1]), OrderedDict(i=Bint[2]))
    x_ = JVP(x)
    with lazy:
        f = x_ + x_

    primal = apply_optimizer(f.primal)
    assert_close(primal, x + x)

    dfdx = f.tangent[str(id(x))](**{str(id(x)): dx})
    assert_close(dfdx, 2*dx)


def test_mul():
    x = random_tensor(OrderedDict(i=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[3]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    dy = random_tensor(OrderedDict(j=Bint[3]))
    x_ = JVP(x)
    y_ = JVP(y)
    with lazy:
        f = x_ * y_

    primal = apply_optimizer(f.primal)
    assert_close(primal, x * y)

    dfdx = f.tangent[str(id(x))](**{str(id(x)): dx})
    assert_close(dfdx, dx*y)

    dfdy = f.tangent[str(id(y))](**{str(id(y)): dy})
    assert_close(dfdy, dy*x)

    # jacfwd
    dx = Tensor(torch.eye(2), OrderedDict(i=Bint[2], l=Bint[2]))
    jacdx = f.tangent[str(id(x))](**{str(id(x)): dx})
    assert_close(jacdx, dx*y)


def test_mul_add():
    x = random_tensor(OrderedDict(i=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[3]))
    z = random_tensor(OrderedDict(k=Bint[4]))
    dx = random_tensor(OrderedDict(i=Bint[2]))
    dy = random_tensor(OrderedDict(j=Bint[3]))
    dz = random_tensor(OrderedDict(k=Bint[4]))
    x_ = JVP(x)
    y_ = JVP(y)
    z_ = JVP(z)
    with lazy:
        f = x_ * y_ + z_

    primal = apply_optimizer(f.primal)
    assert_close(primal, x * y + z)

    dfdx = f.tangent[str(id(x))](**{str(id(x)): dx})
    assert_close(dfdx, dx*y)

    dfdy = f.tangent[str(id(y))](**{str(id(y)): dy})
    # assert_close(dfdy, dy*x+z-z)

    dfdz = f.tangent[str(id(z))](**{str(id(z)): dz})
    breakpoint()
    assert_close(dfdz, dz+x*y-x*y)


def test_reduce_sum():
    x = random_tensor(OrderedDict(j=Bint[4]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    Tx = Variable("dx", Real)
    x_ = Tangent((x, Tx))
    with lazy:
        f, df = x_.reduce(ops.add, "j")
    breakpoint()
    assert_close(apply_optimizer(f), x.reduce(ops.add, "j"))
    assert_close(df(dx=dx), dx.reduce(ops.add, "j"))


def test_reduce_prod():
    x = random_tensor(OrderedDict(j=Bint[4]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    x_ = Tangent((x, dx))
    f, df = x_.reduce(ops.mul, "j")
    assert_close(f, x.reduce(ops.mul, "j"))
    assert_close(df, (f * dx / x).reduce(ops.add, "j"))


def test_reduce_jacfwd():
    x = random_tensor(OrderedDict(j=Bint[4]))
    # dx = Tensor(torch.tensor([1, 0, 0, 0]), OrderedDict(j=Bint[4]))
    dx = Tensor(torch.eye(4), OrderedDict(j=Bint[4], l=Bint[4]))
    x_ = Tangent((x, dx))
    f, df = x_.reduce(ops.mul, "j")
    assert_close(f, x.reduce(ops.mul, "j"))
    assert_close(df, (f * dx / x).reduce(ops.add, "j"))


def test_matmul_tensor():
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    # Tx = Variable("dx", Real)
    Ty = Variable("dy", Reals[4, 5])["j", "k"]
    x_ = Tangent((x, dx))
    y_ = Tangent((y, Ty))
    with lazy:
        xy_ = x_ * y_
        z, dz = xy_.reduce(ops.add, "j")

    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    dy = funsor.terms.Lambda("k", funsor.terms.Lambda("j", dy))
    dz(dy=dy)
    assert_close(apply_optimizer(z), (x * y).reduce(ops.add, "j"))
    assert_close(apply_optimizer(dz), (y * dx + x * dy).reduce(ops.add, "j"))


def test_matmul_jacfwd():
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    dx = Tensor(torch.eye(4), OrderedDict(j=Bint[4], l=Bint[4]))
    dy = Number(0.0)
    x_ = Tangent((x, dx))
    y_ = Tangent((y, dy))
    with lazy:
        xy_ = x_ * y_
        z, dz = xy_.reduce(ops.add, "j")
    assert_close(apply_optimizer(z), (x * y).reduce(ops.add, "j"))
    assert_close(apply_optimizer(dz), (y * dx).reduce(ops.add, "j"))
    assert_close(apply_optimizer(dz), y(j="l").align(tuple(dz.inputs.keys())))


def test_compose():
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    Tx = Variable("dx", Real)
    Ty = Variable("dy", Real)
    x_ = Tangent((x, Tx))
    y_ = Tangent((y, Ty))
    with lazy:
        logx_ = x_.log()
        logxy_ = logx_ * y_
        z, dz = logxy_.reduce(ops.add, "j")

    breakpoint()
    actual_z = apply_optimizer(z)
    expected_z = (x.log() * y).reduce(ops.add, "j")
    assert_close(actual_z, expected_z)
    actual_dz = apply_optimizer(dz(**{"dx": dx, "dy": dy}))
    expected_dz = (y * dx / x + x.log() * dy).reduce(ops.add, "j")
    breakpoint()
    assert_close(actual_dz, expected_dz)
