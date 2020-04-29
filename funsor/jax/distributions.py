# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect

import makefun
import numpyro.distributions as dist

from funsor.distributions import (
    Distribution,
    DistributionMeta,
    FUNSOR_DIST_NAMES,
    backenddist_to_funsor,
    eager_beta,
    eager_binomial,
    eager_categorical_funsor,
    eager_categorical_tensor,
    eager_delta_funsor_funsor,
    eager_delta_funsor_variable,
    eager_delta_tensor,
    eager_delta_variable_variable,
    eager_multinomial,
    eager_mvn,
    eager_normal,
    indepdist_to_funsor,
    maskeddist_to_funsor,
    mvndist_to_funsor,
    transformeddist_to_funsor,
)
from funsor.domains import reals
from funsor.ops import sigmoid
from funsor.tensor import Tensor, dummy_numeric_array
from funsor.terms import Funsor, Variable, eager, to_funsor


################################################################################
# Distribution Wrappers
################################################################################


class _NumPyroWrapper_Categorical(dist.CategoricalProbs):
    pass


def _get_numpyro_dist(dist_name):
    if dist_name in ['Categorical']:
        return globals().get('_NumPyroWrapper_' + dist_name)
    else:
        return getattr(dist, dist_name, None)


def make_dist(backend_dist_class, param_names=()):
    if not param_names:
        param_names = tuple(name for name in inspect.getfullargspec(backend_dist_class.__init__)[0][1:]
                            if name in backend_dist_class.arg_constraints)

    @makefun.with_signature(f"__init__(self, {', '.join(param_names)}, value='value')")
    def dist_init(self, **kwargs):
        return Distribution.__init__(self, *tuple(kwargs[k] for k in self._ast_fields))

    dist_class = DistributionMeta(backend_dist_class.__name__.split("Wrapper_")[-1], (Distribution,), {
        'dist_class': backend_dist_class,
        '__init__': dist_init,
    })

    eager.register(dist_class, *((Tensor,) * (len(param_names) + 1)))(dist_class.eager_log_prob)

    return dist_class


for dist_name, param_names in FUNSOR_DIST_NAMES.items():
    numpyro_dist = _get_numpyro_dist(dist_name)
    if numpyro_dist is not None:
        locals()[dist_name] = make_dist(numpyro_dist, param_names)

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
to_funsor.register(dist.Independent)(indepdist_to_funsor)
to_funsor.register(dist.MaskedDistribution)(maskeddist_to_funsor)
to_funsor.register(dist.TransformedDistribution)(transformeddist_to_funsor)
to_funsor.register(dist.MultivariateNormal)(mvndist_to_funsor)


@to_funsor.register(dist.CategoricalProbs)
# XXX: in Pyro backend, we always convert pyro.distributions.Categorical
# to funsor.distributions.Categorical
@to_funsor.register(dist.CategoricalLogits)
def categorical_to_funsor(numpyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _NumPyroWrapper_Categorical(probs=numpyro_dist.probs)
    return backenddist_to_funsor(new_pyro_dist, output, dim_to_name)


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
