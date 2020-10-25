# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Tuple, Union

import pyro.distributions as dist
import pyro.distributions.testing.fakes as fakes
from pyro.distributions.torch_distribution import MaskedDistribution
import torch

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
    eager_dirichlet_categorical,
    eager_dirichlet_multinomial,
    eager_dirichlet_posterior,
    eager_delta_variable_variable,
    eager_gamma_gamma,
    eager_gamma_poisson,
    eager_multinomial,
    eager_mvn,
    eager_normal,
    indepdist_to_funsor,
    make_dist,
    maskeddist_to_funsor,
    mvndist_to_funsor,
    transformeddist_to_funsor,
)
from funsor.domains import Real, Reals
import funsor.ops as ops
from funsor.tensor import Tensor, dummy_numeric_array
from funsor.terms import Binary, Funsor, Unary, Variable, eager, to_data, to_funsor
from funsor.util import methodof


__all__ = list(x[0] for x in FUNSOR_DIST_NAMES)


################################################################################
# Distribution Wrappers
################################################################################


class _PyroWrapper_BernoulliProbs(dist.Bernoulli):
    def __init__(self, probs, validate_args=None):
        return super().__init__(probs=probs, validate_args=validate_args)

    # XXX: subclasses of Pyro distribution which defines a custom __init__ method
    # should also have `expand` implemented.
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_PyroWrapper_BernoulliProbs, _instance)
        return super().expand(batch_shape, _instance=new)


class _PyroWrapper_BernoulliLogits(dist.Bernoulli):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_PyroWrapper_BernoulliLogits, _instance)
        return super().expand(batch_shape, _instance=new)


class _PyroWrapper_CategoricalLogits(dist.Categorical):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_PyroWrapper_CategoricalLogits, _instance)
        return super().expand(batch_shape, _instance=new)


def _get_pyro_dist(dist_name):
    if dist_name in ['BernoulliProbs', 'BernoulliLogits', 'CategoricalLogits']:
        return globals().get('_PyroWrapper_' + dist_name)
    elif dist_name.startswith('Nonreparameterized'):
        return getattr(fakes, dist_name)
    else:
        return getattr(dist, dist_name)


PYRO_DIST_NAMES = FUNSOR_DIST_NAMES


for dist_name, param_names in PYRO_DIST_NAMES:
    locals()[dist_name] = make_dist(_get_pyro_dist(dist_name), param_names)


# Delta has to be treated specially because of its weird shape inference semantics
@methodof(Delta)  # noqa: F821
@staticmethod
def _infer_value_domain(**kwargs):
    return kwargs['v']


# Multinomial and related dists have dependent Bint dtypes, so we just make them 'real'
# See issue: https://github.com/pyro-ppl/funsor/issues/322
@methodof(Binomial)  # noqa: F821
@methodof(Multinomial)  # noqa: F821
@methodof(DirichletMultinomial)  # noqa: F821
@classmethod
@functools.lru_cache(maxsize=5000)
def _infer_value_domain(cls, **kwargs):
    instance = cls.dist_class(**{k: dummy_numeric_array(domain) for k, domain in kwargs.items()}, validate_args=False)
    return Reals[instance.event_shape]


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


###############################################
# Converting PyTorch Distributions to funsors
###############################################

@to_funsor.register(torch.distributions.Transform)
def transform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    raise NotImplementedError("{} is not a currently supported transform".format(tfm))


@to_funsor.register(torch.distributions.transforms.ExpTransform)
def exptransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    name = next(real_inputs.keys()) if real_inputs else "value"
    return ops.exp(Variable(name, output))


@to_funsor.register(torch.distributions.transforms._InverseTransform)
def inversetransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    expr = to_funsor(tfm._inv, output=output, dim_to_name=dim_to_name, real_inputs=real_inputs)
    assert isinstance(expr, Unary)
    return expr.op.inv(expr.arg)


@to_funsor.register(torch.distributions.transforms.ComposeTransform)
def composetransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    name = next(real_inputs.keys()) if real_inputs else "value"
    expr = Variable(name, output)
    for part in tfm.parts:
        expr = to_funsor(part, output=output, dim_to_name=dim_to_name, real_inputs=real_inputs)(**{name: expr})
    return expr


@to_data.register(Unary[ops.TransformOp, Union[Unary, Variable]])
def transform_to_data(expr, name_to_dim=None):
    raise NotImplementedError("{} is not a currently supported transform".format(expr.op))


@to_data.register(Unary[ops.ExpOp, Union[Unary, Variable]])
def exptransform_to_data(expr, name_to_dim=None):
    tfm = torch.distributions.transforms.ExpTransform()
    if isinstance(expr.arg, Unary):
        tfm = torch.distributions.transforms.ComposeTransform([to_data(expr.arg, name_to_dim=name_to_dim), tfm])
    return tfm


@to_data.register(Unary[ops.LogOp, Union[Unary, Variable]])
def logtransform_to_data(expr, name_to_dim=None):
    tfm = torch.distributions.transforms.ExpTransform().inv
    if isinstance(expr.arg, Unary):
        tfm = torch.distributions.transforms.ComposeTransform([to_data(expr.arg, name_to_dim=name_to_dim), tfm])
    return tfm


to_funsor.register(torch.distributions.Distribution)(backenddist_to_funsor)
to_funsor.register(torch.distributions.Independent)(indepdist_to_funsor)
to_funsor.register(MaskedDistribution)(maskeddist_to_funsor)
to_funsor.register(torch.distributions.TransformedDistribution)(transformeddist_to_funsor)
to_funsor.register(torch.distributions.MultivariateNormal)(mvndist_to_funsor)


@to_funsor.register(torch.distributions.Bernoulli)
def bernoulli_to_funsor(pyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _PyroWrapper_BernoulliLogits(logits=pyro_dist.logits)
    return backenddist_to_funsor(new_pyro_dist, output, dim_to_name)


JointDirichletMultinomial = Contraction[
    Union[ops.LogAddExpOp, ops.NullOp],
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
eager.register(Contraction, ops.LogAddExpOp, ops.AddOp, frozenset, Dirichlet, BernoulliProbs)(  # noqa: F821
    eager_beta_bernoulli)
eager.register(Contraction, ops.LogAddExpOp, ops.AddOp, frozenset, Dirichlet, Categorical)(  # noqa: F821
    eager_dirichlet_categorical)
eager.register(Contraction, ops.LogAddExpOp, ops.AddOp, frozenset, Dirichlet, Multinomial)(  # noqa: F821
    eager_dirichlet_multinomial)
eager.register(Contraction, ops.LogAddExpOp, ops.AddOp, frozenset, Gamma, Gamma)(  # noqa: F821
    eager_gamma_gamma)
eager.register(Contraction, ops.LogAddExpOp, ops.AddOp, frozenset, Gamma, Poisson)(  # noqa: F821
    eager_gamma_poisson)
eager.register(Binary, ops.SubOp, JointDirichletMultinomial, DirichletMultinomial)(  # noqa: F821
    eager_dirichlet_posterior)
