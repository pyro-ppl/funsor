from __future__ import absolute_import, division, print_function

from funsor.delta import Delta, MultiDelta
from funsor.gaussian import Gaussian
from funsor.ops import AddOp, SubOp
from funsor.terms import Binary, Number, eager
from funsor.torch import Tensor


@eager.register(Binary, AddOp, Delta, (Number, Tensor, Gaussian))
def eager_add(op, delta, other):
    if delta.name in other.inputs:
        other = other(**{delta.name: delta.point})
        return op(delta, other)

    return None


@eager.register(Binary, AddOp, (Number, Tensor, Gaussian), Delta)
def eager_add(op, other, delta):
    return delta + other


@eager.register(Binary, SubOp, Delta, Gaussian)
def eager_sub(op, lhs, rhs):
    if lhs.name in rhs.inputs:
        rhs = rhs(**{lhs.name: lhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, SubOp, MultiDelta, Gaussian)
def eager_add_delta_funsor(op, lhs, rhs):
    if lhs.fresh.intersection(rhs.inputs):
        rhs = rhs(**{name: point for name, point in lhs.terms if name in rhs.inputs})
        return op(lhs, rhs)

    return None  # defer to default implementation
