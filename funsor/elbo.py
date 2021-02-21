# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

from multipledispatch.variadic import Variadic

from funsor.cnf import Contraction
from funsor.integrate import Integrate
from funsor.interpretations import StatefulInterpretation
from funsor.ops import AddOp, LogaddexpOp, NullOp
from funsor.terms import Funsor, Reduce

from . import ops


# TODO refactor this once Approximate is merged
class Elbo(StatefulInterpretation):
    """
    Given an approximating ``guide`` funsor, approximates::

        model.reduce(ops.logaddexp, approx_vars)

    by the lower bound::

        Integrate(guide, model - guide, approx_vars)

    :param Funsor guide: A guide or proposal funsor.
    :param frozenset approx_vars: The variables being integrated.
    """

    def __init__(self, guide, approx_vars):
        super().__init__("elbo")
        self.guide = guide
        self.approx_vars = approx_vars


@Elbo.register(Contraction, (LogaddexpOp, NullOp), (AddOp, NullOp), frozenset, tuple)
def elbo_contract(state, sum_op, prod_op, reduced_vars, terms):
    if reduced_vars.isdisjoint(state.approx_vars):
        return None
    if reduced_vars != state.approx_vars:
        raise NotImplementedError("TODO")

    model = reduce(prod_op, terms)
    return Integrate(state.guide, model - state.guide, state.approx_vars)


@Elbo.register(
    Contraction, (LogaddexpOp, NullOp), (AddOp, NullOp), frozenset, Variadic[Funsor]
)
def elbo_contract_variadic(state, sum_op, prod_op, reduced_vars, *terms):
    return elbo_contract(state, sum_op, prod_op, reduced_vars, terms)


@Elbo.register(Reduce, LogaddexpOp, Funsor, frozenset)
def elbo_reduce(state, sum_op, arg, reduced_vars):
    return elbo_contract(state, sum_op, ops.add, reduced_vars, (arg,))


__all__ = [
    "Elbo",
]
