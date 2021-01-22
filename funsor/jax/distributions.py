# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Tuple, Union

import numpyro.distributions as dist

from funsor.cnf import Contraction
from funsor.distribution import (  # noqa: F401
    Bernoulli,
    FUNSOR_DIST_NAMES,
    LogNormal,
    backenddist_to_funsor,
    eager_beta,
    eager_beta_bernoulli,
    eager_binomial,
    eager_categorical_funsor,
    eager_categorical_tensor,
    eager_delta_funsor_funsor,
    eager_delta_funsor_variable,
    eager_delta_tensor,
    eager_delta_variable_variable,
    eager_dirichlet_categorical,
    eager_dirichlet_multinomial,
    eager_dirichlet_posterior,
    eager_gamma_gamma,
    eager_gamma_poisson,
    eager_multinomial,
    eager_mvn,
    eager_normal,
    eager_plate_multinomial,
    expandeddist_to_funsor,
    indepdist_to_funsor,
    make_dist,
    maskeddist_to_funsor,
    transformeddist_to_funsor,
)
from funsor.domains import Real, Reals
import funsor.ops as ops
from funsor.tensor import Tensor
from funsor.terms import Binary, Funsor, Reduce, Variable, eager, to_data, to_funsor
from funsor.util import methodof


################################################################################
# Distribution Wrappers
################################################################################


class _NumPyroWrapper_Binomial(dist.BinomialProbs):
    pass


class _NumPyroWrapper_Categorical(dist.CategoricalProbs):
    pass


class _NumPyroWrapper_Geometric(dist.GeometricProbs):
    pass


class _NumPyroWrapper_Multinomial(dist.MultinomialProbs):
    pass


class _NumPyroWrapper_NonreparameterizedBeta(dist.Beta):
    has_rsample = False


class _NumPyroWrapper_NonreparameterizedDirichlet(dist.Dirichlet):
    has_rsample = False


class _NumPyroWrapper_NonreparameterizedGamma(dist.Gamma):
    has_rsample = False


class _NumPyroWrapper_NonreparameterizedNormal(dist.Normal):
    has_rsample = False


def _get_numpyro_dist(dist_name):
    if dist_name in ['Binomial', 'Categorical', 'Geometric', 'Multinomial'] or \
            dist_name.startswith('Nonreparameterized'):
        return globals().get('_NumPyroWrapper_' + dist_name)
    else:
        return getattr(dist, dist_name, None)


NUMPYRO_DIST_NAMES = FUNSOR_DIST_NAMES
# TODO: remove this after the next NumPyro release
_HAS_RSAMPLE_DISTS = ['Beta', 'Cauchy', 'Chi2', 'Delta', 'Dirichlet', 'Exponential', 'Gamma',
                      'MultivariateNormal', 'Normal', 'Pareto', 'StudentT', 'Uniform']


for dist_name, param_names in NUMPYRO_DIST_NAMES:
    numpyro_dist = _get_numpyro_dist(dist_name)
    if numpyro_dist is not None:
        # TODO: remove this after the next NumPyro release
        if not hasattr(numpyro_dist, "has_rsample"):
            has_rsample = dist_name in _HAS_RSAMPLE_DISTS
            numpyro_dist.has_rsample = has_rsample
            if has_rsample:
                numpyro_dist.rsample = numpyro_dist.sample
        locals()[dist_name] = make_dist(numpyro_dist, param_names)


# Delta has to be treated specially because of its weird shape inference semantics
@methodof(Delta)  # noqa: F821
@staticmethod
def _infer_value_domain(**kwargs):
    return kwargs['v']


@methodof(Categorical)  # noqa: F821
@methodof(CategoricalLogits)  # noqa: F821
@classmethod
def _infer_value_dtype(cls, domains):
    if "logits" in domains:
        return domains["logits"].shape[-1]
    if "probs" in domains:
        return domains["probs"].shape[-1]
    raise ValueError


# Multinomial and related dists have dependent Bint dtypes, so we just make them 'real'
# See issue: https://github.com/pyro-ppl/funsor/issues/322
@methodof(Binomial)  # noqa: F821
@methodof(Multinomial)  # noqa: F821
@classmethod
def _infer_value_dtype(cls, domains):
    return "real"


# TODO fix Delta.arg_constraints["v"] to be a
# constraints.independent[constraints.real]
@methodof(Delta)  # noqa: F821
@staticmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(name, raw_shape):
    if name == "v":
        return Reals[raw_shape]
    elif name == "log_density":
        return Real
    else:
        raise ValueError(name)


# TODO fix Dirichlet.arg_constraints["concentration"] to be a
# constraints.independent[constraints.positive]
@methodof(Dirichlet)  # noqa: F821
@methodof(NonreparameterizedDirichlet)  # noqa: F821
@staticmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(name, raw_shape):
    assert name == "concentration"
    return Reals[raw_shape[-1]]


# TODO: remove this `if` for NumPyro > 0.4.0
if hasattr(dist, "DirichletMultinomial"):
    @methodof(DirichletMultinomial)  # noqa: F821
    @classmethod
    def _infer_value_dtype(cls, domains):
        return "real"

    # TODO fix DirichletMultinomial.arg_constraints["concentration"] to be a
    # constraints.independent[constraints.positive]
    @methodof(DirichletMultinomial)  # noqa: F821
    @classmethod
    @functools.lru_cache(maxsize=5000)
    def _infer_param_domain(cls, name, raw_shape):
        if name == "concentration":
            return Reals[raw_shape[-1]]
        assert name == "total_count"
        return Real


# TODO fix LowRankMultivariateNormal.arg_constraints upstream
@methodof(LowRankMultivariateNormal)  # noqa: F821
@classmethod
@functools.lru_cache(maxsize=5000)
def _infer_param_domain(cls, name, raw_shape):
    if name == "loc":
        return Reals[raw_shape[-1]]
    elif name == "cov_factor":
        return Reals[raw_shape[-2:]]
    elif name == "cov_diag":
        return Reals[raw_shape[-1]]
    raise ValueError(f"{name} invalid param for {cls}")


###########################################################
# Converting distribution funsors to NumPyro distributions
###########################################################

# Convert Delta **distribution** to raw data
@to_data.register(Delta)  # noqa: F821
def deltadist_to_data(funsor_dist, name_to_dim=None):
    v = to_data(funsor_dist.v, name_to_dim=name_to_dim)
    log_density = to_data(funsor_dist.log_density, name_to_dim=name_to_dim)
    return dist.Delta(v, log_density, event_dim=len(funsor_dist.v.output.shape))


###############################################
# Converting NumPyro Distributions to funsors
###############################################

# TODO move these properties upstream to numpyro.distributions
if not hasattr(dist.Independent, "has_rsample"):
    dist.Independent.has_rsample = property(lambda self: self.base_dist.has_rsample)
    dist.Independent.rsample = dist.Independent.sample
if not hasattr(dist.MaskedDistribution, "has_rsample"):
    dist.MaskedDistribution.has_rsample = property(lambda self: self.base_dist.has_rsample)
    dist.MaskedDistribution.rsample = dist.MaskedDistribution.sample
if not hasattr(dist.TransformedDistribution, "has_rsample"):
    dist.TransformedDistribution.has_rsample = property(lambda self: self.base_dist.has_rsample)
    dist.TransformedDistribution.rsample = dist.TransformedDistribution.sample

to_funsor.register(dist.ExpandedDistribution)(expandeddist_to_funsor)
to_funsor.register(dist.Independent)(indepdist_to_funsor)
if hasattr(dist, "MaskedDistribution"):
    to_funsor.register(dist.MaskedDistribution)(maskeddist_to_funsor)
to_funsor.register(dist.TransformedDistribution)(transformeddist_to_funsor)


@to_funsor.register(dist.BinomialProbs)
@to_funsor.register(dist.BinomialLogits)
def categorical_to_funsor(numpyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _NumPyroWrapper_Binomial(total_count=numpyro_dist.total_count, probs=numpyro_dist.probs)
    return backenddist_to_funsor(Binomial, new_pyro_dist, output, dim_to_name)  # noqa: F821


@to_funsor.register(dist.CategoricalProbs)
def categorical_to_funsor(numpyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _NumPyroWrapper_Categorical(probs=numpyro_dist.probs)
    return backenddist_to_funsor(Categorical, new_pyro_dist, output, dim_to_name)  # noqa: F821


@to_funsor.register(dist.GeometricProbs)
def categorical_to_funsor(numpyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _NumPyroWrapper_Geometric(probs=numpyro_dist.probs)
    return backenddist_to_funsor(Geometric, new_pyro_dist, output, dim_to_name)  # noqa: F821


@to_funsor.register(dist.MultinomialProbs)
@to_funsor.register(dist.MultinomialLogits)
def categorical_to_funsor(numpyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _NumPyroWrapper_Multinomial(total_count=numpyro_dist.total_count, probs=numpyro_dist.probs)
    return backenddist_to_funsor(Multinomial, new_pyro_dist, output, dim_to_name)  # noqa: F821


@to_funsor.register(dist.Delta)  # Delta **distribution**
def deltadist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    v = to_funsor(pyro_dist.v, output=Reals[pyro_dist.event_shape], dim_to_name=dim_to_name)
    log_density = to_funsor(pyro_dist.log_density, output=Real, dim_to_name=dim_to_name)
    return Delta(v, log_density)  # noqa: F821


JointDirichletMultinomial = Contraction[
    Union[ops.LogaddexpOp, ops.NullOp],
    ops.AddOp,
    frozenset,
    Tuple[Dirichlet, Multinomial],  # noqa: F821
]


eager.register(Beta, Funsor, Funsor, Funsor)(eager_beta)  # noqa: F821)
eager.register(Binomial, Funsor, Funsor, Funsor)(eager_binomial)  # noqa: F821
eager.register(Multinomial, Tensor, Tensor, Tensor)(eager_multinomial)  # noqa: F821)
eager.register(Categorical, Funsor, Tensor)(eager_categorical_funsor)  # noqa: F821)
eager.register(Categorical, Tensor, Variable)(eager_categorical_tensor)  # noqa: F821)
eager.register(Delta, Tensor, Tensor, Tensor)(eager_delta_tensor)  # noqa: F821
eager.register(Delta, Funsor, Funsor, Variable)(eager_delta_funsor_variable)  # noqa: F821
eager.register(Delta, Variable, Funsor, Variable)(eager_delta_funsor_variable)  # noqa: F821
eager.register(Delta, Variable, Funsor, Funsor)(eager_delta_funsor_funsor)  # noqa: F821
eager.register(Delta, Variable, Variable, Variable)(eager_delta_variable_variable)  # noqa: F821
eager.register(Normal, Funsor, Tensor, Funsor)(eager_normal)  # noqa: F821
eager.register(MultivariateNormal, Funsor, Tensor, Funsor)(eager_mvn)  # noqa: F821
eager.register(Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Dirichlet, BernoulliProbs)(  # noqa: F821
    eager_beta_bernoulli)
eager.register(Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Dirichlet, Categorical)(  # noqa: F821
    eager_dirichlet_categorical)
eager.register(Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Dirichlet, Multinomial)(  # noqa: F821
    eager_dirichlet_multinomial)
eager.register(Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Gamma, Gamma)(  # noqa: F821
    eager_gamma_gamma)
eager.register(Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, Gamma, Poisson)(  # noqa: F821
    eager_gamma_poisson)
if hasattr(dist, "DirichletMultinomial"):
    eager.register(Binary, ops.SubOp, JointDirichletMultinomial, DirichletMultinomial)(  # noqa: F821
        eager_dirichlet_posterior)
eager.register(Reduce, ops.AddOp, Multinomial[Tensor, Funsor, Funsor], frozenset)(  # noqa: F821
    eager_plate_multinomial)

__all__ = list(x[0] for x in FUNSOR_DIST_NAMES if _get_numpyro_dist(x[0]) is not None)
