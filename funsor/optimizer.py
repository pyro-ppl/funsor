# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import collections

from opt_einsum.paths import greedy

import funsor.interpreter as interpreter
from funsor.cnf import Contraction
from funsor.interpretations import (
    DispatchedInterpretation,
    PrioritizedInterpretation,
    eager,
    lazy,
    normalize_base,
)
from funsor.interpreter import get_interpretation
from funsor.ops import DISTRIBUTIVE_OPS, AssociativeOp
from funsor.terms import Funsor
from funsor.typing import Variadic

from . import ops

unfold_base = DispatchedInterpretation()
unfold = PrioritizedInterpretation(unfold_base, normalize_base, lazy)


@unfold.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def unfold_contraction_generic_tuple(red_op, bin_op, reduced_vars, terms):

    for i, v in enumerate(terms):

        if not isinstance(v, Contraction):
            continue

        if v.red_op is ops.null and (v.bin_op, bin_op) in DISTRIBUTIVE_OPS:
            # a * e * (b + c + d) -> (a * e * b) + (a * e * c) + (a * e * d)
            new_terms = tuple(
                Contraction(
                    v.red_op,
                    bin_op,
                    v.reduced_vars,
                    *(terms[:i] + (vt,) + terms[i + 1 :])
                )
                for vt in v.terms
            )
            return Contraction(red_op, v.bin_op, reduced_vars, *new_terms)

        if red_op in (v.red_op, ops.null) and (v.red_op, bin_op) in DISTRIBUTIVE_OPS:
            new_terms = (
                terms[:i]
                + (Contraction(v.red_op, v.bin_op, frozenset(), *v.terms),)
                + terms[i + 1 :]
            )
            return Contraction(v.red_op, bin_op, v.reduced_vars, *new_terms).reduce(
                red_op, reduced_vars
            )

        if v.red_op in (red_op, ops.null) and bin_op in (v.bin_op, ops.null):
            red_op = v.red_op if red_op is ops.null else red_op
            bin_op = v.bin_op if bin_op is ops.null else bin_op
            new_terms = terms[:i] + v.terms + terms[i + 1 :]
            return Contraction(
                red_op, bin_op, reduced_vars | v.reduced_vars, *new_terms
            )

    return None


@unfold.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def unfold_contraction_variadic(r, b, v, *ts):
    return unfold.interpret(Contraction, r, b, v, tuple(ts))


optimize_base = DispatchedInterpretation()
optimize = PrioritizedInterpretation(optimize_base, eager)


# TODO set a better value for this
REAL_SIZE = 3  # the "size" of a real-valued dimension passed to the path optimizer


@optimize.register(
    Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor]
)
def optimize_contraction_variadic(r, b, v, *ts):
    return optimize.interpret(Contraction, r, b, v, tuple(ts))


@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor, Funsor)
@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor)
def eager_contract_base(red_op, bin_op, reduced_vars, *terms):
    return None


@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def optimize_contract_finitary_funsor(red_op, bin_op, reduced_vars, terms):
    if red_op is ops.null or bin_op is ops.null:
        return None
    if (red_op, bin_op) not in DISTRIBUTIVE_OPS:
        return None

    # build opt_einsum optimizer IR
    inputs = [term.input_vars for term in terms]
    size_dict = {
        k: ((REAL_SIZE * v.num_elements) if v.dtype == "real" else v.dtype)
        for term in terms
        for k, v in term.inputs.items()
    }
    outputs = frozenset().union(*inputs) - reduced_vars

    # optimize path with greedy opt_einsum optimizer
    # TODO switch to new 'auto' strategy
    input_names = [frozenset(term.inputs) for term in terms]
    output_names = frozenset(v.name for v in outputs)
    path = greedy(input_names, output_names, size_dict)

    # first prepare a reduce_dim counter to avoid early reduction
    reduce_dim_counter = collections.Counter()
    for input in inputs:
        reduce_dim_counter.update({d: 1 for d in input})

    operands = list(terms)
    for (a, b) in path:
        b, a = tuple(sorted((a, b), reverse=True))
        tb = operands.pop(b)
        ta = operands.pop(a)

        # don't reduce a dimension too early - keep a collections.Counter
        # and only reduce when the dimension is removed from all lhs terms in path
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & ta.input_vars})
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & tb.input_vars})

        # reduce variables that don't appear in other terms
        both_vars = ta.input_vars | tb.input_vars
        path_end_reduced_vars = frozenset(
            d for d in reduced_vars & both_vars if reduce_dim_counter[d] == 0
        )

        # count new appearance of variables that aren't reduced
        reduce_dim_counter.update(
            {d: 1 for d in reduced_vars & (both_vars - path_end_reduced_vars)}
        )

        path_end = Contraction(
            red_op if path_end_reduced_vars else ops.null,
            bin_op,
            path_end_reduced_vars,
            ta,
            tb,
        )
        operands.append(path_end)

    # reduce any remaining dims, if necessary
    final_reduced_vars = (
        frozenset(d for (d, count) in reduce_dim_counter.items() if count > 0)
        & reduced_vars
    )
    if final_reduced_vars:
        path_end = path_end.reduce(red_op, final_reduced_vars)
    return path_end


def apply_optimizer(x):
    with unfold:
        expr = interpreter.reinterpret(x)

    with PrioritizedInterpretation(optimize_base, get_interpretation()):
        return interpreter.reinterpret(expr)
