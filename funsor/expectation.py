from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import funsor.ops as ops
from funsor.domains import reals
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import associate, Finitary
from funsor.terms import Funsor, Unary, eager, monte_carlo, to_funsor, Reduce


class Expectation(Funsor):
    """
    Expectation of an ``integrand`` with resepct to a nonnegative ``measure``.
    """
    def __init__(self, measure, integrand, reduced_vars):
        assert isinstance(measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, Funsor)
        assert measure.output == reals()
        inputs = OrderedDict((k, d) for part in (measure, integrand)
                             for k, d in part.inputs.items()
                             if k not in reduced_vars)
        output = integrand.output
        super(Expectation, self).__init__(inputs, output)
        self.measure = measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def eager_subs(self, subs):
        raise NotImplementedError('TODO')


@eager.register(Expectation, Funsor, Funsor, frozenset)
def eager_expectation(measure, integrand, reduced_vars):
    return (measure * integrand).reduce(ops.add, reduced_vars)


@monte_carlo.register(Expectation, Funsor, Funsor, frozenset)
def monte_carlo_expectation(measure, integrand, reduced_vars):
    if not reduced_vars:
        return measure * integrand

    # Split measure into a finitary product of factors.
    with interpretation(associate):
        integrand = reinterpret(integrand)
    factors = []
    if isinstance(measure, Finitary) and measure.op is ops.mul:
        factors.extend(measure.operands)
    elif (isinstance(measure, Unary) and measure.op is ops.exp and
          isinstance(measure.arg, Finitary) and measure.arg.op is ops.add):
        for operand in measure.arg.operands:
            factors.append(operand.exp())
    else:
        factors.append(measure)
    if len(factors) != len(set(factors)):
        raise NotImplementedError('TODO combine duplicates e.g. x*x -> x**2')

    # Split integrand into a finitary sum of terms.
    with interpretation(associate):
        integrand = reinterpret(integrand)
    terms = []
    if isinstance(integrand, Finitary) and integrand.op is ops.add:
        terms.extend(integrand.operands)
    else:
        terms.append(integrand)
    if len(terms) != len(set(terms)):
        raise NotImplementedError('TODO combine duplicates e.g. x+x -> x*2')

    vars_to_factors = defaultdict(set)
    for factor in factors:
        for var in reduced_vars.intersection(factor.inputs):
            vars_to_factors[var].add(factor)

    # Compute each term separately.
    # TODO share work across terms.
    results = []
    for term in terms:
        term_reduced_vars = reduced_vars.intersection(term.inputs)
        upstream_factors = set().union(*(vars_to_factors[v] for v in term_reduced_vars))
        remaining = reduce(ops.add, set(factors) - upstream_factors, to_funsor(0))
        upstream = reduce(ops.add, upstream_factors, to_funsor(0))
        # Try analytic integration.
        local = eager_expectation(upstream, term, term_reduced_vars)
        if not isinstance(local, Reduce):
            # Fall back to monte carlo integration.
            upstream = upstream.sample(term_reduced_vars)
            local = eager_expectation(upstream, term, term_reduced_vars)
        results.append(eager_expectation(
            remaining, local, reduced_vars - term_reduced_vars))

    return reduce(ops.add, results)


__all__ = [
    'Expectation',
]
