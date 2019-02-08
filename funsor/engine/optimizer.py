"""
Description of the first version of the optimizer:
    1. Rewrite to canonical form of reductions of finitary ops
    2. Rewrite reductions of finitary ops to Contract ops
    3. "De-optimize" by merging as many Contract ops as possible into single Contracts
    4. Optimize by rewriting large contract ops with the greedy path optimizer
    5. Rewrite resulting binary contractions to reductions of binary ops
    6. Rewrite remaining contractions to reductions of finitary ops?
    7. Evaluate?
"""
from __future__ import absolute_import, division, print_function

import funsor.ops as ops
from funsor.terms import Binary, Finitary, Funsor, Reduction, Tensor, Unitary

from .paths import greedy


@engine.handle(Unary, Binary)
def binary_to_finitary(op, lhs, rhs=None):
    """convert Binary/Unary to Finitary"""
    return Finitary(op, [lhs, rhs] if rhs is not None else [lhs])


@engine.handle(Reduce)
def finitary_to_contract(op, arg, reduce_dims):
    """convert Reduce(Finitary/Binary) to Contract"""
    if isinstance(arg, Finitary):
        return Contract(*arg.terms, reduce_dims=reduce_dims)


@engine.handle(Contract)
def deoptimize(x, backend=None):
    """
    Convert multiple Contract ops to a single, larger Contract
    """
    assert isinstance(x, Funsor)

    if backend is None:
        backend = x.backend

    if isinstance(x, Contract)


@engine.handle(Contract)
def optimize_path(op, x, optimizer="greedy"):
    r"""
    Recursively convert large Contract ops to many smaller binary Contracts
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
        raise NotImplementedError("{} not a valid optimizer".format(optimizer))

    # convert path back to sequence of Contracts
    operands = x.operands[:]
    for (a, b) in path:
        operands.pop(b)
        path_end = Contract(a, b, backend=x)
        operands[a] = path_end

    return path_end


@engine.handle(Contract)
def contract_to_finitary(op, x):
    """convert Contract to Reduce(Finitary/Binary)"""
    if isinstance(x, Contract):
        return Reduce(op, Finitary(x), reduce_dims=x.reduce_dims)
    return x


def finitary_to_binary(op, x):
    """convert Finitary to Binary/Unary when appropriate"""
    raise NotImplementedError


def apply_optimizer(x):
    # apply passes
    x = binary_to_finitary(x)
    x = finitary_to_contract(x)
    x = deoptimize(x)
    x = optimize_path(x)
    x = contract_to_finitary(x)
    x = finitary_to_binary(x)
    return x
