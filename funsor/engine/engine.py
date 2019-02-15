from __future__ import absolute_import, division, print_function

from six.moves import reduce
from funsor.six import singledispatch

import funsor.distributions as dist
from funsor.handlers import OpRegistry, effectful
from funsor.terms import Binary, Finitary, Funsor, Number, Reduction, Substitution, Unary, Variable
from funsor.torch import Arange, Tensor


class EagerEval(OpRegistry):
    _terms_processed = {}
    _terms_postprocessed = {}


@EagerEval.register(Tensor)
def eager_tensor(dims, data):
    return Tensor(dims, data).materialize()  # .data


@EagerEval.register(Number)
def eager_number(data, dtype):
    return Number(data, dtype)


# TODO add general Normal
@EagerEval.register(dist.Normal)
def eager_distribution(loc, scale, value):
    return dist.Normal(loc, scale, value=value).materialize()


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
    return Substitution(arg, subs).materialize()


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


@singledispatch
def eval(x):
    r"""
    Overloaded partial evaluation of deferred expression.
    Default semantics: do nothing (reflect)

    This handles a limited class of expressions, raising
    ``NotImplementedError`` in unhandled cases.

    :param Funsor x: An input funsor, typically deferred.
    :return: An evaluated funsor.
    :rtype: Funsor
    :raises: NotImplementedError
    """
    assert isinstance(x, Funsor)
    raise NotImplementedError


@eval.register(Tensor)
def _eval_tensor(x):
    # return effectful(Tensor, Tensor)(x.dims, x.data)
    return Tensor(x.dims, x.data)


@eval.register(dist.Distribution)
def _eval_distribution(x):
    # return effectful(type(x), type(x))(**{k: eval(v) for k, v in x.params.items()})
    return type(x)(**{k: eval(v) for k, v in x.params.items()})


@eval.register(Number)
def _eval_number(x):
    # return effectful(Number, Number)(x.data, type(x.data))
    return Number(x.data, type(x.data))


@eval.register(Variable)
def _eval_variable(x):
    # return effectful(Variable, Variable)(x.name, x.shape[0])
    return Variable(x.name, x.shape[0])


@eval.register(Substitution)
def _eval_substitution(x):
    # return effectful(Substitution, Substitution)(
    return Substitution(
        eval(x.arg),
        tuple((dim, eval(value)) for (dim, value) in x.subs)
    )


@eval.register(Unary)
def _eval_unary(x):
    # return effectful(Unary, Unary)(x.op, eval(x.v))
    return Unary(x.op, eval(x.v))


@eval.register(Binary)
def _eval_binary(x):
    # return effectful(Binary, Binary)(x.op, eval(x.lhs), eval(x.rhs))
    return Binary(x.op, eval(x.lhs), eval(x.rhs))


@eval.register(Finitary)
def _eval_finitary(x):
    # return effectful(Finitary, Finitary)(x.op, tuple(eval(tx) for tx in x.operands))
    return Finitary(x.op, tuple(eval(tx) for tx in x.operands))


@eval.register(Reduction)
def _eval_reduction(x):
    # return effectful(Reduction, Reduction)(x.op, eval(x.arg), x.reduce_dims)
    return Reduction(x.op, eval(x.arg), x.reduce_dims)


__all__ = [
    'eval',
    'EagerEval',
]
