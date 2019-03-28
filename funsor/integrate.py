from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.contract import Contract
from funsor.terms import Funsor, Reduce, eager


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
        super(Integrate, self).__init__(inputs, output)
        self.log_measure = log_measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def eager_subs(self, subs):
        raise NotImplementedError('TODO')


def _simplify_integrate(fn, log_measure, integrand, reduced_vars):
    """
    Reduce free variables that do not appear in both inputs.
    """
    if not reduced_vars:
        return log_measure.exp() * integrand

    log_measure_vars = frozenset(log_measure.inputs)
    integrand_vars = frozenset(integrand.inputs)
    assert reduced_vars <= log_measure_vars | integrand_vars
    progress = False
    if not reduced_vars <= log_measure_vars:
        integrand = integrand.reduce(ops.add, reduced_vars - log_measure_vars)
        reduced_vars = reduced_vars & log_measure_vars
        progress = True
    if not reduced_vars <= integrand_vars:
        log_measure = log_measure.reduce(ops.logaddexp, reduced_vars - integrand_vars)
        reduced_vars = reduced_vars & integrand_vars
        progress = True
    if progress:
        return Integrate(log_measure, integrand, reduced_vars)

    return fn(log_measure, integrand, reduced_vars)


def integrator(fn):
    """
    Decorator for integration implementations.
    """
    fn = interpreter.debug_logged(fn)
    return functools.partial(_simplify_integrate, fn)


@eager.register(Integrate, Funsor, Funsor, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    return Contract(ops.add, ops.mul, log_measure.exp(), integrand, reduced_vars)


@eager.register(Integrate, Reduce, Funsor, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    if log_measure.op is ops.logaddexp:
        if not log_measure.reduced_vars.isdisjoint(reduced_vars):
            raise NotImplementedError('TODO alpha convert')
        arg = Integrate(log_measure.arg, integrand, reduced_vars)
        return arg.reduce(ops.add, log_measure.reduced_vars)

    return Contract(ops.add, ops.mul, log_measure.exp(), integrand, reduced_vars)


__all__ = [
    'Integrate',
    'integrator',
]
