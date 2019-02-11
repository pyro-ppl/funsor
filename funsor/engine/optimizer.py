"""
Description of the first version of the optimizer:
    1. Rewrite to canonical form of reductions of finitary ops
    2. Rewrite reductions of finitary ops to Contract ops
    3. "De-optimize" by merging as many Contract ops as possible into single Contracts
    4. Optimize by rewriting large contract ops with the greedy opt_einsum path optimizer
"""
from __future__ import absolute_import, division, print_function

import collections

from funsor.handlers import OpRegistry
from funsor.terms import Binary, Finitary, Reduction, Unary

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
    # two cases to rewrite, which we handle in separate branches:
    if all(isinstance(term, (Finitary,)) for term in terms):  # TODO check distributivity
        # Case 1) Finitary(Finitary) -> Finitary
        new_terms = []
        for term in terms:
            if isinstance(term, Finitary) and term.op == op:
                new_terms.extend(term.terms)
            else:
                new_terms.append(term)

        return Finitary(op, new_terms)
    elif all(isinstance(term, Reduction) for term in terms):  # TODO check distributivity
        # Case 2) Finitary(Reduction, Reduction) -> Reduction(Finitary(lhs.arg, rhs.arg))
        new_terms = []
        new_reduce_dims = set()
        for term in terms:
            new_terms.append(term.arg)
            new_reduce_dims = new_reduce_dims.union(term.reduce_dims)
        return Reduction(terms[0].op, Finitary(op, new_terms), new_reduce_dims)
    elif all(not isinstance(term, (Reduction, Finitary)) for term in terms):
        return Finitary(op, terms)  # nothing to do, reflect
    else:
        # Note: if we can't rewrite all terms in the finitary, fail for now
        # A more sophisticated strategy is to apply this rule recursively
        # Alternatively, we could do this rewrite on Binary ops instead of Finitary
        raise NotImplementedError("TODO(eb8680) handle mixed case")


@Deoptimize.register(Reduction)
def deoptimize_reduction(op, arg, reduce_dims):
    """
    Rewrite to the largest possible Reduction(Finitary) by combining Reductions
    Assumes that all input Reduction/Finitary ops have been rewritten
    """
    # one case to rewrite:
    if isinstance(arg, Reduction) and arg.op == op:
        # Reduction(Reduction) -> Reduction
        new_reduce_dims = reduce_dims.union(arg.reduce_dims)
        return Reduction(op, arg.arg, new_reduce_dims)
    else:  # nothing to do, reflect
        return Reduction(op, arg, reduce_dims)


class Optimize(OpRegistry):
    pass


@Optimize.register(Reduction)  # TODO need Finitary as well?
def optimize_reduction(op, arg, reduce_dims):
    r"""
    Recursively convert large Reduce(Finitary) ops to many smaller versions
    by reordering execution with a modified opt_einsum optimizer
    """
    if not isinstance(arg, Finitary):  # nothing to do, reflect
        return Reduction(op, arg, reduce_dims)

    # build opt_einsum optimizer IR
    inputs = []
    size_dict = {}
    for term in arg.terms:
        inputs.append(frozenset(d for d in term.dims))
        # TODO get sizes right
        size_dict.update({d: 2 for d in term.dims})
    outputs = frozenset().union(*inputs) - reduce_dims

    # optimize path with greedy opt_einsum optimizer
    path = greedy(inputs, outputs, size_dict, cost_fn='memory-removed')

    # convert path IR back to sequence of Reduction(Finitary(...))

    # first prepare a reduce_dim counter to avoid early reduction
    reduce_dim_counter = collections.Counter()
    for input in inputs:
        reduce_dim_counter.update((d, 1) for d in input)

    reduce_op, finitary_op = op, arg.op
    operands = arg.terms[:]
    for (a, b) in path:
        operands.pop(b)
        path_end_finitary = Finitary(finitary_op, [a, b])

        # don't reduce a dimension too early - keep a collections.Counter
        # and only reduce when the dimension is removed from all lhs terms in path
        reduce_dim_counter.subtract((d, 1) for d in reduce_dims & a.dims)
        reduce_dim_counter.subtract((d, 1) for d in reduce_dims & b.dims)

        path_end_reduce_dims = frozenset(d for d in reduce_dims & (a.dims | b.dims)
                                         if reduce_dim_counter[d] == 0)

        path_end = Reduction(reduce_op, path_end_finitary, path_end_reduce_dims)
        operands[a] = path_end

    # reduce any remaining dims, if necessary
    final_reduce_dims = frozenset(d for (d, count) in reduce_dim_counter.items()
                                  if count > 0)
    if final_reduce_dims:
        path_end = Reduction(reduce_op, path_end, final_reduce_dims)
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
