from multipledispatch import dispatch

from funsor.ops import AddOp, ExpOp, LogOp, MulOp, NegOp
from funsor.terms import Binary, Unary


def invert(x, y, name="value"):
    result = _invert(name, x, y)
    if result is None:
        raise NotImplementedError(
            f"Failed to solve for {name} in {x} == {y}")
    return result


@dispatch
def _invert(name, x, y):
    if isinstance(x, Variable) and x.name == name:
        return y
    raise NotImplementedError(
        f"Failed to solve for {name} in {x} == {y}")


invert.register = _invert.register


@invert.register(str, Unary[ggg], Funsor)
def _(name, x, y):
    return invert(name, x.arg, -y)


@invert.register(str, Unary[ExpOp], Funsor)
def _(name, x, y):
    return invert(name, x.arg, ops.log(y))


@invert.register(str, Unary[LogOp], Funsor)
def _(name, x, y):
    return invert(name, x.arg, ops.exp(y))


@invert.register(str, Binary[AddOp, Funsor, Funsor], Funsor)
def _(name, x, y):
    if name not in x.lhs.inputs:
        return invert(name, x.lhs, y - x.rhs)
    if name not in x.rhs.inputs:
        return invert(name, x.arg, y - x.lhs)
    # TODO handle affine combinations


@invert.register(str, Binary[MulOp, Funsor, Funsor], Funsor)
def _(name, x, y):
    if name not in x.lhs.inputs:
        return invert(name, x.lhs, y / x.rhs)
    if name not in x.rhs.inputs:
        return invert(name, x.arg, y / x.lhs)
    # TODO handle affine combinations
