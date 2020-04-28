# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import pyro.distributions as dist
import pyro.distributions.testing.fakes as fakes
from pyro.distributions.torch_distribution import MaskedDistribution
import torch

from funsor.distributions import (
    FUNSOR_DIST_NAMES,
    backenddist_to_funsor,
    indepdist_to_funsor,
    make_backend_dist,
    maskeddist_to_funsor,
    mvndist_to_funsor,
    transformeddist_to_funsor,
)
from funsor.domains import reals
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


def _get_pyro_dist(dist_name):
    if dist_name in ['BernoulliProbs', 'BernoulliLogits', 'CategoricalLogits']:
        return locals()['_PyroWrapper_' + dist_name]
    elif dist_name.startswith('Nonreparameterized'):
        return getattr(fakes, dist_name)
    else:
        return getattr(dist, dist_name)


for dist_name, param_names in FUNSOR_DIST_NAMES.items():
    locals()[dist_name] = make_backend_dist(_get_pyro_dist(dist_name), param_names)

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

to_funsor.register(dist.Distribution)(backenddist_to_funsor)
to_funsor.register(dist.Indepdent)(indepdist_to_funsor)
to_funsor.register(MaskedDistribution)(maskeddist_to_funsor)
to_funsor.register(dist.TransformedDistribution)(transformeddist_to_funsor)
to_funsor.register(dist.MultivariateNormal)(mvndist_to_funsor)


@to_funsor.register(torch.distributions.Bernoulli)
def bernoulli_to_funsor(pyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _PyroWrapper_BernoulliLogits(logits=pyro_dist.logits)
    return backenddist_to_funsor(new_pyro_dist, output, dim_to_name)
