# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import OrderedDict

import pyro.distributions as dist
import pyro.distributions.testing.fakes as fakes
from pyro.distributions.torch_distribution import MaskedDistribution
import torch

import funsor.delta
from funsor.distributions import make_dist
from funsor.domains import reals
from funsor.gaussian import Gaussian
from funsor.tensor import dummy_numeric_array
from funsor.terms import to_funsor


################################################################################
# Distribution Wrappers
################################################################################


class _PyroWrapper_BernoulliProbs(dist.Bernoulli):
    def __init__(self, probs, validate_args=None):
        return super().__init__(probs=probs, validate_args=validate_args)


class _PyroWrapper_BernoulliLogits(dist.Bernoulli):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


class _PyroWrapper_CategoricalLogits(dist.Categorical):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


_wrapped_pyro_dists = [
    (dist.Beta, ()),
    (_PyroWrapper_BernoulliProbs, ('probs',)),
    (_PyroWrapper_BernoulliLogits, ('logits',)),
    (dist.Binomial, ('total_count', 'probs')),
    (dist.Multinomial, ('total_count', 'probs')),
    (dist.Categorical, ('probs',)),
    (_PyroWrapper_CategoricalLogits, ('logits',)),
    (dist.Poisson, ()),
    (dist.Gamma, ()),
    (dist.VonMises, ()),
    (dist.Dirichlet, ()),
    (dist.DirichletMultinomial, ()),
    (dist.Normal, ()),
    (dist.MultivariateNormal, ('loc', 'scale_tril')),
    (dist.Delta, ()),
    (fakes.NonreparameterizedBeta, ()),
    (fakes.NonreparameterizedGamma, ()),
    (fakes.NonreparameterizedNormal, ()),
    (fakes.NonreparameterizedDirichlet, ()),
]

for pyro_dist_class, param_names in _wrapped_pyro_dists:
    locals()[pyro_dist_class.__name__.split("_PyroWrapper_")[-1].split(".")[-1]] = \
        make_dist(pyro_dist_class, param_names)

# Delta has to be treated specially because of its weird shape inference semantics
Delta._infer_value_domain = classmethod(lambda cls, **kwargs: kwargs['v'])  # noqa: F821


# Multinomial and related dists have dependent bint dtypes, so we just make them 'real'
# See issue: https://github.com/pyro-ppl/funsor/issues/322
@functools.lru_cache(maxsize=5000)
def _multinomial_infer_value_domain(cls, **kwargs):
    instance = cls.dist_class(**{k: dummy_numeric_array(domain) for k, domain in kwargs.items()}, validate_args=False)
    return reals(*instance.event_shape)


Binomial._infer_value_domain = classmethod(_multinomial_infer_value_domain)  # noqa: F821
Multinomial._infer_value_domain = classmethod(_multinomial_infer_value_domain)  # noqa: F821
DirichletMultinomial._infer_value_domain = classmethod(_multinomial_infer_value_domain)  # noqa: F821


###############################################
# Converting PyTorch Distributions to funsors
###############################################

@to_funsor.register(torch.distributions.Distribution)
def torchdistribution_to_funsor(pyro_dist, output=None, dim_to_name=None):
    import funsor.torch.distributions  # TODO find a better way to do this lookup
    funsor_dist_class = getattr(funsor.torch.distributions, type(pyro_dist).__name__.split("_PyroWrapper_")[-1])
    params = [to_funsor(
            getattr(pyro_dist, param_name),
            output=funsor_dist_class._infer_param_domain(
                param_name, getattr(getattr(pyro_dist, param_name), "shape", ())),
            dim_to_name=dim_to_name)
        for param_name in funsor_dist_class._ast_fields if param_name != 'value']
    return funsor_dist_class(*params)


@to_funsor.register(torch.distributions.Independent)
def indepdist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    dim_to_name = OrderedDict((dim - pyro_dist.reinterpreted_batch_ndims, name)
                              for dim, name in dim_to_name.items())
    dim_to_name.update(OrderedDict((i, f"_pyro_event_dim_{i}") for i in range(-pyro_dist.reinterpreted_batch_ndims, 0)))
    result = to_funsor(pyro_dist.base_dist, dim_to_name=dim_to_name)
    for i in reversed(range(-pyro_dist.reinterpreted_batch_ndims, 0)):
        name = f"_pyro_event_dim_{i}"
        result = funsor.terms.Independent(result, "value", name, "value")
    return result


@to_funsor.register(MaskedDistribution)
def maskeddist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    mask = to_funsor(pyro_dist._mask.float(), output=output, dim_to_name=dim_to_name)
    funsor_base_dist = to_funsor(pyro_dist.base_dist, output=output, dim_to_name=dim_to_name)
    return mask * funsor_base_dist


@to_funsor.register(torch.distributions.Bernoulli)
def bernoulli_to_funsor(pyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _PyroWrapper_BernoulliLogits(logits=pyro_dist.logits)
    return torchdistribution_to_funsor(new_pyro_dist, output, dim_to_name)


@to_funsor.register(torch.distributions.TransformedDistribution)
def transformeddist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    raise NotImplementedError("TODO implement conversion of TransformedDistribution")


@to_funsor.register(torch.distributions.MultivariateNormal)
def torchmvn_to_funsor(pyro_dist, output=None, dim_to_name=None, real_inputs=OrderedDict()):
    funsor_dist = torchdistribution_to_funsor(pyro_dist, output=output, dim_to_name=dim_to_name)
    if len(real_inputs) == 0:
        return funsor_dist
    discrete, gaussian = funsor_dist(value="value").terms
    inputs = OrderedDict((k, v) for k, v in gaussian.inputs.items() if v.dtype != 'real')
    inputs.update(real_inputs)
    return discrete + Gaussian(gaussian.info_vec, gaussian.precision, inputs)
