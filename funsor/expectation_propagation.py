# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

from multipledispatch.variadic import Variadic

from funsor.adjoint import AdjointTape
from funsor.cnf import Contraction
from funsor.interpreter import dispatched_interpretation, interpretation
from funsor.ops import DISTRIBUTIVE_OPS, PRODUCT_INVERSES, UNITS, AssociativeOp
from funsor.terms import Funsor, eager, lazy, moment_matching, normalize, reflect, to_funsor


@dispatched_interpretation
def message_passing(cls, *args):
    result = message_passing.dispatch(cls, *args)(*args)
    if result is None:
        result = eager.dispatch(cls, *args)(*args)
    if result is None:
        result = normalize.dispatch(cls, *args)(*args)
    if result is None:
        result = reflect(cls, *args)
    return result


@message_passing.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def message_passing_contract_default(*args):
    return Contraction(args[0], args[1], args[2], tuple(args[3:]))


# @message_passing.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def ep_generic_contraction_adjoint(red_op, bin_op, reduced_vars, terms):
    """Generic message-passing algorithm with projection step"""
    if (red_op, bin_op) not in DISTRIBUTIVE_OPS or not reduced_vars:
        return None
    approxs = [to_funsor(UNITS[bin_op]) for term in terms]  # initialize approximations to 1
    for step in range(10):  # fixpoint iteration
        with interpretation(moment_matching), AdjointTape() as tape:
            ws = [PRODUCT_INVERSES[bin_op](term, approx) for term, approx in zip(terms, approxs)]
            approx = reduce(bin_op, approxs)
            step_reduced_vars = reduced_vars.intersection(approx.inputs)
            result = reduce(bin_op, [bin_op(approx, w).reduce(red_op, step_reduced_vars.union(w.inputs)) for w in ws])
        with interpretation(moment_matching):  # at step 0 this should end up projecting the prior terms
            approxs = tuple(tape.adjoint(red_op, bin_op, result, ws).values())
    return result


@message_passing.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def ep_generic_contraction_no_adjoint(red_op, bin_op, reduced_vars, terms):
    """Generic message-passing algorithm with projection step"""
    if (red_op, bin_op) not in DISTRIBUTIVE_OPS or not reduced_vars:
        return None
    scalar_terms = [term for term in terms if not term.inputs]
    terms = [term for term in terms if term.inputs]
    approxs = [to_funsor(UNITS[bin_op]) for term in terms]  # initialize approximations to 1
    for step in range(10):  # fixpoint iteration
        with interpretation(normalize)
            full_approx = reduce(bin_op, approxs)
        with interpretation(moment_matching):
            for i, (term, approx) in enumerate(zip(terms, list(approxs))):
                w = PRODUCT_INVERSES[bin_op](term, approx)
                step_reduced_vars = reduced_vars.intersection(full_approx.inputs)
                approxs[i] = bin_op(full_approx, w).reduce(red_op, step_reduced_vars.union(w.inputs) - frozenset(term.inputs))
                approxs[i] = PRODUCT_INVERSES[bin_op](approxs[i], approxs[i].reduce(red_op))
                full_approx = bin_op(full_approx, PRODUCT_INVERSES[bin_op](approxs[i], approx))

    ws = [PRODUCT_INVERSES[bin_op](term, approx) for term, approx in zip(terms, approxs)]
    approx = reduce(bin_op, approxs)
    step_reduced_vars = reduced_vars.intersection(approx.inputs)
    result = reduce(bin_op, scalar_terms + [bin_op(approx, w).reduce(red_op, step_reduced_vars.union(w.inputs)) for w in ws])
    return result
