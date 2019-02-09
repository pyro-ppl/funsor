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
from funsor.handlers import OpRegistry

from .paths import greedy


class Desugar(OpRegistry):
    pass


@Desugar.register(Unary, Binary)
def binary_to_finitary(op, lhs, rhs=None):
    """convert Binary/Unary to Finitary"""
    return Finitary(op, [lhs, rhs] if rhs is not None else [lhs])


class Deoptimize(OpRegistry):
    pass


@Deoptimize.register(Finitary)
def deoptimize_finitary(op, terms):
    raise NotImplementedError("TODO")


@Deoptimize.register(Reduce)
def deoptimize_reduce(op, arg, reduce_dims):
    raise NotImplementedError("TODO")


class Optimize(OpRegistry):
    pass


@Optimize.register(Reduce)
def optimize_path(op, arg, reduce_dims):
    r"""
    Recursively convert large Contract ops to many smaller binary Contracts
    by reordering execution with a modified opt_einsum optimizer
    """
    if not isinstance(arg, Finitary):
        return Reduce(op, arg, reduce_dims)

    # build opt_einsum optimizer IR
    inputs = []
    size_dict = {}
    for term in terms:
        inputs.append(set(d for d in term.dims))
        # TODO get sizes right
        size_dict.update({d: 2 for d in term.dims})
    outputs = set().union(*inputs) - reduce_dims

    # optimize path
    path = greedy(inputs, output, size_dict, cost_fn='memory-removed')

    # convert path IR back to sequence of Reduce(..., Finitary(...))
    reduce_op, finitary_op = op, arg.op
    operands = x.operands[:]
    for (a, b) in path:
        operands.pop(b)
        # TODO don't reduce a dimension too early - keep a collections.Counter
        path_end = Reduce(reduce_op, Finitary(finitary_op, [a, b]),
                          reduce_dims & a.dims & b.dims)
        operands[a] = path_end

    return path_end


class Resugar(OpRegistry):
    pass

# @Resugar.register(Finitary)
# def finitary_to_binary(op, x):
#     """convert Finitary to Binary/Unary when appropriate"""
#     raise NotImplementedError


def apply_optimizer(x):
    # apply passes
    with Desugar():
        x = eval(x)

    with Deoptimize():
        x = eval(x)

    with Optimize():
        x = eval(x)

    # with Resugar():
    #     x = eval(x)

    return x
