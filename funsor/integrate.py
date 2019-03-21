from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import funsor.ops as ops
from funsor.contract import Contract
from funsor.terms import Funsor, eager


class Integrate(Funsor):
    def __init__(self, log_measure, integrand, reduced_vars):
        assert isinstance(log_measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, frozenset)
        inputs = OrderedDict((k, d) for term in (log_measure, integrand)
                             for (k, d) in term.inputs.items()
                             if k not in reduced_vars)
        output = integrand.output
        super(Integrate, self).__init__(inputs, output)
        self.log_measure = log_measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def eager_subs(self, subs):
        raise NotImplementedError('TODO')


def simplify_integrate(log_measure, integrand, reduced_vars):
    """
    Reduce free variables that do not appear in both inputs.
    """
    log_measure_vars = frozenset(log_measure.inputs)
    integrand_vars = frozenset(integrand.inputs)
    assert reduced_vars <= log_measure_vars | integrand_vars
    progress = False
    if not reduced_vars <= log_measure_vars:
        integrand = integrand.reduce(ops.logaddexp, reduced_vars - log_measure_vars)
        reduced_vars = reduced_vars & log_measure_vars
        progress = True
    if not reduced_vars <= integrand_vars:
        log_measure = log_measure.reduce(ops.add, reduced_vars - integrand_vars)
        reduced_vars = reduced_vars & integrand_vars
        progress = True

    if progress:
        return Integrate(log_measure, integrand, reduced_vars)

    return None


@eager.register(Integrate, Funsor, Funsor, frozenset)
def eager_integrate(log_measure, integrand, reduced_vars):
    result = simplify_integrate(log_measure, integrand, reduced_vars)
    if result is not None:
        return result

    return Contract(log_measure.exp(), integrand, reduced_vars)


__all__ = [
    'Integrate',
    'simplify_integrate',
]
