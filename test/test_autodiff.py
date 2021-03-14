# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.autodiff import JVP, to_var, to_arg, fjit, linearize, grad
from funsor.testing import assert_close, random_tensor
from funsor.terms import Variable, Number, lazy, Lambda, Binary, Funsor
from funsor.tensor import Tensor
from funsor.optimizer import apply_optimizer
from funsor.interpreter import interpretation
from funsor.factory import make_funsor, Bound, Fresh, Has
from funsor.sum_product import MarkovProduct


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
    x_ = JVP((x, Tx))
    with lazy:
        f, df = x_.reduce(ops.add, "j")
    breakpoint()
    assert_close(apply_optimizer(f), x.reduce(ops.add, "j"))
    assert_close(df(dx=dx), dx.reduce(ops.add, "j"))


def test_reduce_prod():
    x = random_tensor(OrderedDict(j=Bint[4]))
    dx = random_tensor(OrderedDict(j=Bint[4]))
    x_ = JVP((x, dx))
    f, df = x_.reduce(ops.mul, "j")
    assert_close(f, x.reduce(ops.mul, "j"))
    assert_close(df, (f * dx / x).reduce(ops.add, "j"))


def test_reduce_jacfwd():
    x = random_tensor(OrderedDict(j=Bint[4]))
    # dx = Tensor(torch.tensor([1, 0, 0, 0]), OrderedDict(j=Bint[4]))
    dx = Tensor(torch.eye(4), OrderedDict(j=Bint[4], l=Bint[4]))
    x_ = JVP((x, dx))
    f, df = x_.reduce(ops.mul, "j")
    assert_close(f, x.reduce(ops.mul, "j"))
    assert_close(df, (f * dx / x).reduce(ops.add, "j"))


@make_funsor
def MatMul(
        a: Has[{"i"}],
        b: Has[{"i"}],
        i: Bound
    ) -> Fresh[lambda a: a]:
    return Prod(a, b).reduce(ops.add, i)

@make_funsor
def Prod(
        x: Funsor,
        y: Funsor
    ) -> Fresh[lambda x: x]:
    return x * y


def test_fjit():
    # Product
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    cProd = fjit(Prod, x, y)

    x2 = random_tensor(OrderedDict(j=Bint[4]))
    y2 = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    expected = Prod(x2, y2)
    actual = cProd(x=to_arg(x2), y=to_arg(y2))
    assert_close(actual, expected)

    # MarkovProduct
    trans = random_tensor(OrderedDict(time=Bint[5], prev=Bint[3], curr=Bint[3]))
    cMarkovProduct = fjit(MarkovProduct, ops.logaddexp, ops.add, trans, "time", {"prev": "curr"})

    trans2 = random_tensor(OrderedDict(time=Bint[5], prev=Bint[3], curr=Bint[3]))
    expected = MarkovProduct(ops.logaddexp, ops.add, trans2, "time", {"prev": "curr"})
    actual = cMarkovProduct(trans=to_arg(trans2))
    assert_close(actual, expected)


def test_grad():
    # Add
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    result = grad(Binary, ops.add, x, y, log=False)
    breakpoint()

    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    expected = dx + dy
    actual = linearAdd(lhs=to_arg(dx), rhs=to_arg(dy))
    assert_close(actual, expected)
    assert_close(z, x + y)


def test_linearize():
    # Add
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    (z, linearAdd), linear_vars = linearize(Binary, ops.add, x, y, log=False)

    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    expected = dx + dy
    actual = linearAdd(lhs=to_arg(dx), rhs=to_arg(dy))
    assert_close(actual, expected)
    assert_close(z, x + y)

    # Add in a LogFunctor
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    with funsor.terms.lazy:
        z, linearAdd = linearize(Binary, ops.add, x, y, log=True)

    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    expected = ops.logaddexp(ops.add(y, dx), ops.add(x, dy))
    breakpoint()
    actual = linearAdd(lhs=to_arg(dx), rhs=to_arg(dy))
    assert_close(actual, expected)

    # MarkovProduct in a LogFunctor
    trans = random_tensor(OrderedDict(time=Bint[5], prev=Bint[3], curr=Bint[3]))
    with funsor.terms.lazy:
        z, linearMP = linearize(MarkovProduct, ops.logaddexp, ops.add, trans, "time", {"prev": "curr"}, log=True)

    dtrans = random_tensor(OrderedDict(time=Bint[5], prev=Bint[3], curr=Bint[3]))
    # expected = MarkovProduct(ops.logaddexp, ops.add, trans2, "time", {"prev": "curr"})
    actual = linearMP(trans=to_arg(dtrans))
    # assert_close(actual, expected)


def test_transpose():
    # Mul
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    z, linearAdd = linearize(Binary, ops.mul, x, y, log=False)
    linear_transpose(linearAdd, {"lhs", "rhs"}, log=False)

    # Add
    x = random_tensor(OrderedDict(j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    z, linearAdd = linearize(Binary, ops.add, x, y, log=False)
    linear_transpose(linearAdd, {"lhs", "rhs"}, log=False)

    dx = random_tensor(OrderedDict(j=Bint[4]))
    dy = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    expected = dx + dy
    actual = linearAdd(dlhs=to_arg(dx), drhs=to_arg(dy))
    assert_close(actual, expected)
    assert_close(z, x + y)
