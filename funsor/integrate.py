# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Union

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.gaussian import Gaussian, align_gaussian, _mv, _trace_mm, _vv
from funsor.tensor import Tensor
from funsor.terms import (
    Funsor,
    FunsorMeta,
    Number,
    Subs,
    Unary,
    Variable,
    _convert_reduced_vars,
    eager,
    normalize,
    substitute,
    to_funsor
)


class IntegrateMeta(FunsorMeta):
    """
    Wrapper to convert reduced_vars arg to a frozenset of str.
    """
    def __call__(cls, log_measure, integrand, reduced_vars):
        inputs = log_measure.inputs.copy()
        inputs.update(integrand.inputs)
        reduced_vars = _convert_reduced_vars(reduced_vars, inputs)
        return super().__call__(log_measure, integrand, reduced_vars)


class Integrate(Funsor, metaclass=IntegrateMeta):
    """
    Funsor representing an integral wrt a log density funsor.

    :param Funsor log_measure: A log density funsor treated as a measure.
    :param Funsor integrand: An integrand funsor.
    :param reduced_vars: An input name or set of names to reduce.
    :type reduced_vars: str, Variable, or set or frozenset thereof.
    """
    def __init__(self, log_measure, integrand, reduced_vars):
        assert isinstance(log_measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, Variable) for v in reduced_vars)
        reduced_names = frozenset(v.name for v in reduced_vars)
        inputs = OrderedDict((k, d) for term in (log_measure, integrand)
                             for (k, d) in term.inputs.items()
                             if k not in reduced_names)
        output = integrand.output
        fresh = frozenset()
        bound = {v.name: v.output for v in reduced_vars}
        super(Integrate, self).__init__(inputs, output, fresh, bound)
        self.log_measure = log_measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def _alpha_convert(self, alpha_subs):
        assert set(self.bound).issuperset(alpha_subs)
        reduced_vars = frozenset(Variable(alpha_subs.get(v.name, v.name), v.output)
                                 for v in self.reduced_vars)
        alpha_subs = {k: to_funsor(v, self.integrand.inputs.get(k, self.log_measure.inputs.get(k)))
                      for k, v in alpha_subs.items()}
        log_measure = substitute(self.log_measure, alpha_subs)
        integrand = substitute(self.integrand, alpha_subs)
        return log_measure, integrand, reduced_vars


@normalize.register(Integrate, Funsor, Funsor, frozenset)
def normalize_integrate(log_measure, integrand, reduced_vars):
    return Contraction(ops.add, ops.mul, reduced_vars, log_measure.exp(), integrand)


@normalize.register(Integrate,
                    Contraction[Union[ops.NullOp, ops.LogaddexpOp], ops.AddOp, frozenset, tuple],
                    Funsor, frozenset)
def normalize_integrate_contraction(log_measure, integrand, reduced_vars):
    reduced_names = frozenset(v.name for v in reduced_vars)
    delta_terms = [t for t in log_measure.terms if isinstance(t, Delta)
                   and t.fresh.intersection(reduced_names, integrand.inputs)]
    for delta in delta_terms:
        integrand = integrand(**{name: point for name, (point, log_density) in delta.terms
                                 if name in reduced_names.intersection(integrand.inputs)})
    return normalize_integrate(log_measure, integrand, reduced_vars)


@eager.register(Contraction, ops.AddOp, ops.MulOp, frozenset,
                Unary[ops.ExpOp, Union[GaussianMixture, Delta, Gaussian, Number, Tensor]],
                (Variable, Delta, Gaussian, Number, Tensor, GaussianMixture))
def eager_contraction_binary_to_integrate(red_op, bin_op, reduced_vars, lhs, rhs):
    reduced_names = frozenset(v.name for v in reduced_vars)
    if not (reduced_names.issubset(lhs.inputs) and reduced_names.issubset(rhs.inputs)):
        args = red_op, bin_op, reduced_vars, (lhs, rhs)
        result = eager.dispatch(Contraction, *args)(*args)
        if result is not None:
            return result

    args = lhs.log(), rhs, reduced_vars
    result = eager.dispatch(Integrate, *args)(*args)
    if result is not None:
        return result

    return None


@eager.register(Integrate, GaussianMixture, Funsor, frozenset)
def eager_integrate_gaussianmixture(log_measure, integrand, reduced_vars):
    real_vars = frozenset(v for v in reduced_vars if v.dtype == "real")
    if reduced_vars <= real_vars:
        discrete, gaussian = log_measure.terms
        return discrete.exp() * Integrate(gaussian, integrand, reduced_vars)
    return None


########################################
# Delta patterns
########################################

@eager.register(Integrate, Delta, Funsor, frozenset)
def eager_integrate(delta, integrand, reduced_vars):
    delta_fresh = frozenset(Variable(k, delta.inputs[k]) for k in delta.fresh)
    if reduced_vars.isdisjoint(delta_fresh):
        return None
    reduced_names = frozenset(v.name for v in reduced_vars)
    subs = tuple((name, point) for name, (point, log_density) in delta.terms
                 if name in reduced_names)
    new_integrand = Subs(integrand, subs)
    new_log_measure = Subs(delta, subs)
    result = Integrate(new_log_measure, new_integrand, reduced_vars - delta_fresh)
    return result


########################################
# Gaussian patterns
########################################

@eager.register(Integrate, Gaussian, Variable, frozenset)
def eager_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(v for v in reduced_vars if v.dtype == "real")
    if real_vars == frozenset([integrand]):
        loc = ops.cholesky_solve(ops.unsqueeze(log_measure.info_vec, -1), log_measure._precision_chol).squeeze(-1)
        data = loc * ops.unsqueeze(ops.exp(log_measure.log_normalizer.data), -1)
        data = data.reshape(loc.shape[:-1] + integrand.output.shape)
        inputs = OrderedDict((k, d) for k, d in log_measure.inputs.items() if d.dtype != 'real')
        result = Tensor(data, inputs)
        return result.reduce(ops.add, reduced_vars - real_vars)
    return None  # defer to default implementation


@eager.register(Integrate, Gaussian, Gaussian, frozenset)
def eager_integrate(log_measure, integrand, reduced_vars):
    reduced_names = frozenset(v.name for v in reduced_vars)
    real_vars = frozenset(v.name for v in reduced_vars if v.dtype == "real")
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
            norm = ops.exp(lhs.log_normalizer.data)
            lhs_cov = ops.cholesky_inverse(lhs._precision_chol)
            lhs_loc = ops.cholesky_solve(ops.unsqueeze(lhs.info_vec, -1), lhs._precision_chol).squeeze(-1)
            vmv_term = _vv(lhs_loc, rhs_info_vec - 0.5 * _mv(rhs_precision, lhs_loc))
            data = norm * (vmv_term - 0.5 * _trace_mm(rhs_precision, lhs_cov))
            inputs = OrderedDict((k, d) for k, d in inputs.items() if k not in reduced_names)
            result = Tensor(data, inputs)
            return result.reduce(ops.add, reduced_names - real_vars)

        raise NotImplementedError('TODO implement partial integration')

    return None  # defer to default implementation


__all__ = [
    'Integrate',
]
