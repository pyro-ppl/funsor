from __future__ import absolute_import, division, print_function

from six.moves import reduce

from funsor.handlers import OpRegistry
from funsor.engine.interpreter import eval
from funsor.distributions import Normal
from funsor.handlers import OpRegistry
from funsor.terms import Binary, Finitary, Number, Reduction, Substitution, Unary, Variable
from funsor.torch import Arange, Tensor, LazyCall


class EagerEval(OpRegistry):
    pass


# TODO add general Distribution
@EagerEval.register(Normal)
def eager_distribution(loc, scale, value):
    return Normal(loc, scale)(value=value)


# TODO separate materialization from eager evaluation
@EagerEval.register(Variable)
def eager_variable(name, size):
    if isinstance(size, int):
        return Arange(name, size)
    else:
        return Variable(name, size)


@EagerEval.register(Unary)
def eager_unary(op, v):
    return op(v)


@EagerEval.register(Substitution)
def eager_substitution(arg, subs):  # this is the key...
    return arg(**dict(subs))


@EagerEval.register(Binary)
def eager_binary(op, lhs, rhs):
    return op(lhs, rhs)


@EagerEval.register(Finitary)
def eager_finitary(op, operands):
    if len(operands) == 1:
        return eager_unary(op, operands[0])  # XXX is this necessary?
    return reduce(op, operands[1:], operands[0])


@EagerEval.register(Reduction)
def eager_reduce(op, arg, reduce_dims):
    return arg.reduce(op, reduce_dims)


@EagerEval.register(LazyCall)
def eager_reduce(fn, args):
    if all(isinstance(x, (Number, Tensor)) for x in args):
        return fn(*args)
    return LazyCall(fn, args)


class Materialize(OpRegistry):
    pass


@Materialize.register(Variable)
def _materialize_variable(name, size):
    if isinstance(size, int):
        return Arange(name, size)
    else:
        return Variable(name, size)


def materialize(x):
    x = Materialize(eval)(x)
    x = EagerEval(eval)(x)
    return x


__all__ = [
    'materialize',
]
