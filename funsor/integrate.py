from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from funsor.terms import Funsor, eager
from funsor.contract import Contract


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


@eager.register(Integrate, Funsor, Funsor, frozenset)
def eager_integrate(log_measure, integrand, reduced_vars):
    return Contract(log_measure.exp(), integrand, reduced_vars)


__all__ = [
    'Integrate',
]
