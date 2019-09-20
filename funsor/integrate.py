from collections import OrderedDict

from funsor.terms import Funsor, to_funsor


class Integrate(Funsor):
    """
    Funsor representing an integral wrt a log density funsor.
    """
    def __init__(self, log_measure, integrand, reduced_vars):
        assert isinstance(log_measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, frozenset)
        inputs = OrderedDict((k, d) for term in (log_measure, integrand)
                             for (k, d) in term.inputs.items()
                             if k not in reduced_vars)
        output = integrand.output
        fresh = frozenset()
        bound = reduced_vars
        super(Integrate, self).__init__(inputs, output, fresh, bound)
        self.log_measure = log_measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def _alpha_convert(self, alpha_subs):
        reduced_vars = frozenset(alpha_subs.get(k, k) for k in self.reduced_vars)
        bound_types = self.log_measure.inputs.copy()
        bound_types.update(self.integrand.inputs)
        alpha_subs = {k: to_funsor(v, bound_types[k]) for k, v in alpha_subs.items()}
        log_measure, integrand, _ = super()._alpha_convert(alpha_subs)
        return log_measure, integrand, reduced_vars


__all__ = [
    'Integrate',
]
