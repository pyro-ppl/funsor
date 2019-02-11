from __future__ import absolute_import, division, print_function

import opt_einsum

import funsor.ops as ops
from funsor.terms import Arange, Binary, Contract, Finitary, Funsor, Reduction, Tensor, Unary, Variable

from funsor.handlers import default_handler, effectful, OpRegistry


class EagerEval(OpRegistry):
    pass


@EagerEval.register(Tensor)
def eager_tensor(x):
    return x.materialize()


@EagerEval.register(Variable)
def eager_variable(name, size):
    if isinstance(size, int):
        return Arange(name, size)
    else:
        return Variable(name, size)


@EagerEval.register(Unary)
def eager_unary(op, v):
    return op(v)


@EagerEval.register(Substitute)
def eager_substitute(arg, subs):  # this is the key...
    for (dim, value) in subs:
        pass
    raise ValueError("FIXME")


@EagerEval.register(Binary)
def eager_binary(op, lhs, rhs):
    return op(lhs, rhs)


@EagerEval.register(Finitary)
def eager_finitary(op, terms):
    if len(terms) == 1:
        return eager_unary(op, terms[0])  # XXX is this necessary?
    return reduce(op, terms[1:], terms[0])


@EagerEval.register(Reduction)
def eager_reduce(op, arg, reduce_dims):
    assert isinstance(op, Tensor)  # XXX is this true?
    return arg.reduce(op, reduce_dims)


# @default_handler(EagerEval())
def eval(x):  # TODO get input args right
    r"""
    Overloaded partial evaluation of deferred expression.
    Default semantics: do nothing

    This handles a limited class of expressions, raising
    ``NotImplementedError`` in unhandled cases.

    :param Funsor x: An input funsor, typically deferred.
    :return: An evaluated funsor.
    :rtype: Funsor
    :raises: NotImplementedError
    """
    assert isinstance(x, Funsor)

    if isinstance(x, Tensor):
        return effectful(Tensor)(Tensor)(x.dims, x.data)

    if isinstance(x, Variable):
        return effectful(Variable)(Variable)(x.name, x.size)

    if isinstance(x, Substitute):
        return effectful(Substitute)(Substitute)(
            eval(x.arg),
            tuple((dim, eval(value)) for (dim, value) in x.subs)
        )
   
    # Arithmetic operations
    if isinstance(x, Unary):
        return effectful(Unary)(Unary)(x.op, eval(x.v))

    if isinstance(x, Binary):
        return effectful(Binary)(Binary)(x.op, eval(x.lhs), eval(x.rhs))

    if isinstance(x, Finitary):
        return effectful(Finitary)(Finitary)(x.op, [eval(tx) for tx in x.terms])

    # Reductions
    if isinstance(x, Reduction):
        return effectful(Reduction)(Reduction)(x.op, eval(x.arg), x.reduce_dims)

    raise NotImplementedError


__all__ = [
    'eval',
    'EagerEval',
]
