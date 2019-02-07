from __future__ import absolute_import, division, print_function

import opt_einsum

import funsor.ops as ops
from funsor.terms import Binary, Finitary, Funsor, Reduction, Tensor, Unitary

from .paths import greedy


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


def canonicalize(op, x):
    """
    parse a bunch of Binary and Reduction to Finitary and Reduction canonical form
    so we can use the optimizer to turn it into a better Binary and Reduction

    Rewrite rules:
        Reduce(Binary) -> Reduce(Finitary)
    """
    pending = [x]
    terms = []
    final = None
    while pending:
        x = pending.pop()
        if isinstance(x, Reduction):
            pass
        elif isinstance(x, Binary):
            pass

    return final


def to_finitary(op, x):
    """convert Binary/Unary to Finitary"""
    pass


def from_finitary(op, x):
    """convert Finitary to Binary/Unary when appropriate"""
    pass


def to_contract(op, x):
    """convert Reduce(Finitary/Binary) to Contract"""
    pass


def from_contract(op, x):
    """convert Contract to Reduce(Finitary/Binary)"""
    pass


def optimize(op, x, optimizer="greedy"):
    r"""
    Take a bunch of Finitary and Reduction and turn them to Binary and Reduction path
    by reordering execution with a modified opt_einsum optimizer

    Rewrite rules (type only):
        Reduce(Finitary) -> Reduce(Binary(x, Reduce(Binary(y, ...))))
    """
    # extract path
    if isinstance(x, Reduction):
        x.arg, x.reduce_dims = *x

    # optimize path
    if optimizer == "greedy":
        path = greedy(inputs, output, size_dict, cost_fn='memory-removed')
    else:
        raise NotImplementedError

    # TODO repopulate path

    path = fuse(x)

    return path


def eval(x, optimize=True):
    r"""
    Optimized evaluation of deferred expressions.

    This handles a limited class of expressions, raising
    ``NotImplementedError`` in unhandled cases.

    :param Funsor x: An input funsor, typically deferred.
    :return: An evaluated funsor.
    :rtype: Funsor
    :raises: NotImplementedError
    """
    assert isinstance(x, Funsor)
    if optimize:
        x = canonicalize(x)
        x = optimize(x, optimizer="greedy")

    # evaluate the path
    head = x
    pending = []
    while not _finished(head):
        head = pending.pop()
        if isinstance(head, Tensor):
            head = head
       
        if isinstance(head, Reduction):
            head = head.reduce(head.op, eval(head.terms, optimize=optimize))

        if isinstance(head, Binary):
            head = head.op(eval(head.lhs, optimize=optimize), eval(head.rhs, optimize=optimize))

        if isinstance(head, Unary):
            head = head.op(eval(head.v, optimize=optimize))

        if isinstance(head, Finitary):
            head = reduce(head.op, [eval(thead, optimize=optimize) for thead in head.terms])

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

    raise NotImplementedError


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
    data = opt_einsum.contract(*args, **kwargs)
    return Tensor(dims, data)


__all__ = [
    'contract',
    'eval',
]
