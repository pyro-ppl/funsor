"""
Description of the first version of the optimizer:
    1. Rewrite to canonical form of reductions of finitary ops
    2. Rewrite reductions of finitary ops to Contract ops
    3. "De-optimize" by merging as many Contract ops as possible into single Contracts
    4. Optimize by rewriting large contract ops with the greedy path optimizer
"""
from __future__ import absolute_import, division, print_function

from collections import Counter

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
    """
    Rewrite to the largest possible Finitary(Finitary/Reduction) by moving Reductions
    Assumes that all input Finitary ops have been rewritten
    """
    # two cases to rewrite:
    # 1) Finitary(Reduction, Reduction) -> Reduction(Finitary(lhs.arg, rhs.arg))
    # 2) Finitary(Finitary) -> Finitary
    new_terms = []
    for term in terms:
        if isinstance(term, Finitary) and term.op == op:
            new_terms.extend(term.terms)
        # elif isinstance(term, Reduction):
        #     pass
        else:
            new_terms.append(term)
    return Finitary(op, new_terms)


@Deoptimize.register(Reduction)
def deoptimize_reduce(op, arg, reduce_dims):
    """
    Rewrite to the largest possible Reduction(Finitary) by combining Reductions
    Assumes that all input Reduction/Finitary ops have been rewritten
    """
    # example of something to rewrite:
    # Reduction(Reduction) -> Reduction
    if isinstance(arg, Reduction) and arg.op == op:
        new_reduce_dims = reduce_dims.union(arg.reduce_dims)
        return Reduction(op, arg.arg, new_reduce_dims)
    else:
        return Reduction(op, arg, reduce_dims)


class Optimize(OpRegistry):
    pass


@Optimize.register(Reduction)  # TODO need Finitary as well?
def optimize_path(op, arg, reduce_dims):
    r"""
    Recursively convert large Reduce(Finitary) ops to many smaller versions
    by reordering execution with a modified opt_einsum optimizer
    """
    if not isinstance(arg, Finitary):  # reflect
        return Reduction(op, arg, reduce_dims)

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

    # convert path IR back to sequence of Reduction(..., Finitary(...))
    reduce_dim_counter = collections.Counter()
    reduce_op, finitary_op = op, arg.op
    operands = x.operands[:]
    for (a, b) in path:
        operands.pop(b)
        path_end_finitary = Finitary(finitary_op, [a, b])

        # TODO don't reduce a dimension too early - keep a collections.Counter
        # and only reduce when the dimension is removed from all lhs terms in the path
        path_end_reduce_dims = reduce_dims & a.dims & b.dims

        path_end = Reduction(reduce_op, path_end_finitary, path_end_reduce_dims)
        operands[a] = path_end

    return path_end


def apply_optimizer(x):

    # TODO can any of these be combined into a single eval?
    with Desugar():
        x = eval(x)

    with Deoptimize():
        x = eval(x)

    with Optimize():
        x = eval(x)

    return x
