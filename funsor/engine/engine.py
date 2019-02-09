from __future__ import absolute_import, division, print_function

import opt_einsum

import funsor.ops as ops
from funsor.terms import Binary, Contract, Finitary, Funsor, Reduction, Tensor, Unitary

from .handlers import effectful, EvalPass
from .paths import greedy


class EagerEval(EvalPass):
    pass


@EagerEval.register(Tensor)
def eager_tensor(x):
    return x.materialize()


@EagerEval.register(Variable)
def eager_variable(name, size):
    if isinstance(size, int):
        return Arange(name, size)  # TODO get this call right
    else:
        return Variable(name, size)


@EagerEval.register(Unary)
def eager_unary(op, v):
    return op(v)


@EagerEval.register(Substitute)
def eager_substitute(arg, subs):
    # TODO
    raise NotImplementedError("TODO")


@EagerEval.register(Binary)
def eager_binary(op, lhs, rhs):
    return op(lhs, rhs)


@EagerEval.register(Finitary)
def eager_finitary(op, terms):
    return reduce(op, terms[1:], terms[0])


@EagerEval.register(Reduce)
def eager_reduce(op, arg, reduce_dims):
    raise NotImplementedError("TODO")  # TODO implement


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

    # evaluate the path
    if isinstance(x, Tensor):
        return effectful(Tensor)(x)

    if isinstance(x, Variable):
        return effectful(Variable)()

    if isinstance(x, Substitute):
        return effectful(Substitute)(...)  # TODO get args right
   
    if isinstance(x, Unary):
        return effectful(Unary)(x.op, eval(x.v))

    if isinstance(x, Binary):
        return effectful(Binary)(x.op, eval(x.lhs), eval(x.rhs))

    if isinstance(x, Finitary):
        return effectful(Finitary)(x.op, [eval(tx) for tx in x.terms])

    if isinstance(x, Reduce):
        return effectful(Reduce)(x.op, eval(x.arg), x.reduce_dims)

    raise NotImplementedError


############
# old
############


def _parse_reduction(op, x):
    if isinstance(x, Reduction) and x.op is op:
        yield x.arg, x.reduce_dims


def _parse_commutative(op, x):
    pending = [x]
    terms = []
    while pending:
        x = pending.pop()
        if isinstance(x, Binary) and x.op is op:
            pending.append(x.lhs)
            pending.append(x.rhs)
        else:
            terms.append(x)
    return terms


def contract(*operands, **kwargs):
    r"""
    Sum-product contraction operation.

    :param tuple dims: a tuple of strings of output dimensions. Any input dim
        not requested as an output dim will be summed out.
    :param \*operands: multiple :class:`Funsor`s.
    :param tuple dims: An optional tuple of output dims to preserve.
        Defaults to ``()``, meaning all dims are contracted.
    :param str backend: An opt_einsum backend, defaults to 'torch'.
    :return: A contracted funsor.
    :rtype: Funsor
    """
    # # Handle sum-product contractions.
    # for arg, reduce_dims in _parse_reduction(ops.add, x):
    #     operands = _parse_commutative(ops.mul, arg)
    #     dims = tuple(d for d in arg.dims if d not in reduce_dims)
    #     head = contract(*operands, dims=dims)

    # # Handle log-sum-product-exp contractions.
    # for arg, reduce_dims in _parse_reduction(ops.logaddexp, x):
    #     operands = _parse_commutative(ops.add, arg)
    #     dims = tuple(d for d in arg.dims if d not in reduce_dims)
    #     return contract(*operands, dims=dims, backend='pyro.ops.einsum.torch_log')


    assert all(isinstance(x, Funsor) for x in operands)
    dims = kwargs.pop('dims', ())
    assert isinstance(dims, tuple)
    assert all(isinstance(d, str) for d in dims)
    kwargs.setdefault('backend', 'torch')
    args = []
    for x in operands:
        x = x.materialize()
        if not isinstance(x, Tensor):
            raise NotImplementedError
        args.extend([x.data, x.dims])
    args.append(dims)
    data = opt_einsum.contract(*args, **kwargs)  # TODO use backend einsum directly
    return Tensor(dims, data)


__all__ = [
    'contract',
    'eval',
]
