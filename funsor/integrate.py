import functools
from collections import OrderedDict
from functools import reduce
from typing import Union

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.gaussian import Gaussian, _mv, _trace_mm, _vv, align_gaussian, cholesky_inverse, cholesky_solve
from funsor.terms import Funsor, Subs, Variable, eager, normalize, substitute, to_funsor
from funsor.torch import Tensor


class Integrate(Funsor):
    """
    Funsor representing an integral wrt a log density funsor.
    """
    def __init__(self, log_measure, integrand, reduced_vars):
        assert isinstance(log_measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, str) for v in reduced_vars)
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
        assert self.bound.issuperset(alpha_subs)
        reduced_vars = frozenset(alpha_subs.get(k, k) for k in self.reduced_vars)
        alpha_subs = {k: to_funsor(v, self.integrand.inputs.get(k, self.log_measure.inputs.get(k)))
                      for k, v in alpha_subs.items()}
        log_measure = substitute(self.log_measure, alpha_subs)
        integrand = substitute(self.integrand, alpha_subs)
        return log_measure, integrand, reduced_vars


@normalize.register(Integrate, Funsor, Funsor, frozenset)
def normalize_integrate(log_measure, integrand, reduced_vars):
    # Reduce free variables that do not appear in both inputs.
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

    return None  # defer to default implementation


@normalize.register(Integrate,
                    Contraction[Union[ops.NullOp, ops.LogAddExpOp], ops.AddOp, frozenset, tuple],
                    Funsor, frozenset)
def normalize_integrate_contraction(log_measure, integrand, reduced_vars):
    delta_terms = [t for t in log_measure.terms if isinstance(t, Delta)
                   and t.fresh.intersection(reduced_vars, integrand.inputs)]
    for delta in delta_terms:
        integrand = integrand(**{name: point for name, (point, log_density) in delta.terms
                                 if name in reduced_vars.intersection(integrand.inputs)})
    return normalize_integrate(log_measure, integrand, reduced_vars)


# This is a hack to treat normalize as higher priority than eager.
def integrator(fn):

    @functools.wraps(fn)
    def wrapped(log_measure, integrand, reduced_vars):
        if not (reduced_vars.issubset(log_measure.inputs) and
                reduced_vars.issubset(integrand.inputs)):
            return None  # normalize
        return fn(log_measure, integrand, reduced_vars)

    return wrapped


@eager.register(Integrate,
                (Funsor, Delta, Contraction[ops.NullOp, ops.AddOp, frozenset, tuple]),
                Contraction[Union[ops.NullOp, ops.LogAddExpOp], ops.AddOp, frozenset, tuple], frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    terms = [Integrate(log_measure, term, reduced_vars)
             for term in integrand.terms]
    return reduce(ops.add, terms)


@eager.register(Integrate,
                Contraction[Union[ops.NullOp, ops.LogAddExpOp], ops.AddOp, frozenset, tuple],
                Funsor, frozenset)
@integrator
def eager_integrate_const(log_measure, integrand, reduced_vars):
    if reduced_vars:
        consts, factors = [], []
        for term in log_measure.terms:
            (consts if reduced_vars.isdisjoint(term.inputs) else factors).append(term)
        if consts:
            sum_vars = log_measure.reduced_vars
            assert sum_vars.isdisjoint(integrand.inputs)
            const = reduce(ops.add, consts)
            log_measure = reduce(ops.add, factors)
            inner = Integrate(log_measure, integrand, reduced_vars)
            return Integrate(const, inner, sum_vars)

    return None  # defer to default implementation


@eager.register(Integrate, GaussianMixture, Funsor, frozenset)
@integrator
def eager_integrate_gaussianmixture(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if reduced_vars <= real_vars:
        discrete, gaussian = log_measure.terms
        return discrete.exp() * Integrate(gaussian, integrand, reduced_vars)
    return None


########################################
# Tensor patterns
########################################

@eager.register(Integrate, Tensor,
                (Funsor, Contraction[Union[ops.NullOp, ops.LogAddExpOp], ops.AddOp, frozenset, tuple]),
                frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    return Contraction(ops.add, ops.mul, reduced_vars, log_measure.exp(), integrand)


########################################
# Delta patterns
########################################

@eager.register(Integrate, Delta, Funsor, frozenset)
@integrator
def eager_integrate(delta, integrand, reduced_vars):
    if delta.fresh.isdisjoint(reduced_vars):
        return None
    subs = tuple((name, point) for name, (point, log_density) in delta.terms
                 if name in reduced_vars)
    new_integrand = Subs(integrand, subs)
    new_log_measure = Subs(delta, subs)
    result = Integrate(new_log_measure, new_integrand, reduced_vars - delta.fresh)
    return result


########################################
# Gaussian patterns
########################################

@eager.register(Integrate, Gaussian, Variable, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if real_vars == frozenset([integrand.name]):
        loc = cholesky_solve(log_measure.info_vec.unsqueeze(-1), log_measure._precision_chol).squeeze(-1)
        data = loc * log_measure.log_normalizer.data.exp().unsqueeze(-1)
        data = data.reshape(loc.shape[:-1] + integrand.output.shape)
        inputs = OrderedDict((k, d) for k, d in log_measure.inputs.items() if d.dtype != 'real')
        result = Tensor(data, inputs)
        return result.reduce(ops.add, reduced_vars - real_vars)
    return None  # defer to default implementation


@eager.register(Integrate, Gaussian, Gaussian, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if real_vars:

        lhs_reals = frozenset(k for k, d in log_measure.inputs.items() if d.dtype == 'real')
        rhs_reals = frozenset(k for k, d in integrand.inputs.items() if d.dtype == 'real')
        if lhs_reals == real_vars and rhs_reals <= real_vars:
            inputs = OrderedDict((k, d) for t in (log_measure, integrand)
                                 for k, d in t.inputs.items())
            lhs_info_vec, lhs_precision = align_gaussian(inputs, log_measure)
            rhs_info_vec, rhs_precision = align_gaussian(inputs, integrand)
            lhs = Gaussian(lhs_info_vec, lhs_precision, inputs)

            # Compute the expectation of a non-normalized quadratic form.
            # See "The Matrix Cookbook" (November 15, 2012) ss. 8.2.2 eq. 380.
            # http://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
            norm = lhs.log_normalizer.data.exp()
            lhs_cov = cholesky_inverse(lhs._precision_chol)
            lhs_loc = cholesky_solve(lhs.info_vec.unsqueeze(-1), lhs._precision_chol).squeeze(-1)
            vmv_term = _vv(lhs_loc, rhs_info_vec - 0.5 * _mv(rhs_precision, lhs_loc))
            data = norm * (vmv_term - 0.5 * _trace_mm(rhs_precision, lhs_cov))
            inputs = OrderedDict((k, d) for k, d in inputs.items() if k not in reduced_vars)
            result = Tensor(data, inputs)
            return result.reduce(ops.add, reduced_vars - real_vars)

        raise NotImplementedError('TODO implement partial integration')

    return None  # defer to default implementation


__all__ = [
    'Integrate',
]
