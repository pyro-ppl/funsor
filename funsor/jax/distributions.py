# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import numpyro.distributions as dist

from funsor.distributions import (
    backenddist_to_funsor,
    indepdist_to_funsor,
    make_dist,
    mvndist_to_funsor,
    transformeddist_to_funsor,
)
from funsor.domains import reals
from funsor.tensor import dummy_numeric_array
from funsor.terms import to_funsor


################################################################################
# Distribution Wrappers
################################################################################


class _NumPyroWrapper_Categorical(dist.CategoricalProbs):
    pass


_wrapped_backend_dists = [
    (dist.Beta, ()),
    (dist.BernoulliProbs, ('probs',)),
    (dist.BernoulliLogits, ('logits',)),
    (dist.Binomial, ('total_count', 'probs')),
    (dist.Multinomial, ('total_count', 'probs')),
    (_NumPyroWrapper_Categorical, ('probs',)),
    (dist.CategoricalLogits, ('logits',)),
    (dist.Poisson, ()),
    (dist.Gamma, ()),
    (dist.Dirichlet, ()),
    (dist.Normal, ()),
    (dist.MultivariateNormal, ('loc', 'scale_tril')),
    (dist.Delta, ()),
]

for backend_dist_class, param_names in _wrapped_backend_dists:
    locals()[backend_dist_class.__name__.split("Wrapper_")[-1]] = make_dist(backend_dist_class, param_names)

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


###############################################
# Converting PyTorch Distributions to funsors
###############################################

to_funsor.register(dist.Distribution)(backenddist_to_funsor)
to_funsor.register(dist.Indepdent)(indepdist_to_funsor)
# TODO: register MaskedDistribution
to_funsor.register(dist.TransformedDistribution)(transformeddist_to_funsor)
to_funsor.register(dist.MultivariateNormal)(mvndist_to_funsor)


@to_funsor.register(dist.CategoricalProbs)
def categorical_to_funsor(numpyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _NumPyroWrapper_Categorical(probs=numpyro_dist.probs)
    return backenddist_to_funsor(new_pyro_dist, output, dim_to_name)
