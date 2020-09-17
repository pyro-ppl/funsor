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
    make_dist,
    maskeddist_to_funsor,
    mvndist_to_funsor,
    transformeddist_to_funsor,
)
from funsor.domains import Reals
import funsor.ops as ops
from funsor.tensor import Tensor, dummy_numeric_array
from funsor.terms import Binary, Funsor, Unary, Variable, eager, to_funsor


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


PYRO_DIST_NAMES = FUNSOR_DIST_NAMES + [
    ('DirichletMultinomial', ('concentration', 'total_count')),
    ('VonMises', ('loc', 'concentration')),
]


for dist_name, param_names in PYRO_DIST_NAMES:
    locals()[dist_name] = make_dist(_get_pyro_dist(dist_name), param_names)


# Delta has to be treated specially because of its weird shape inference semantics
Delta._infer_value_domain = classmethod(lambda cls, **kwargs: kwargs['v'])  # noqa: F821


# Multinomial and related dists have dependent Bint dtypes, so we just make them 'real'
# See issue: https://github.com/pyro-ppl/funsor/issues/322
@functools.lru_cache(maxsize=5000)
def _multinomial_infer_value_domain(cls, **kwargs):
    instance = cls.dist_class(**{k: dummy_numeric_array(domain) for k, domain in kwargs.items()}, validate_args=False)
    return Reals[instance.event_shape]


Binomial._infer_value_domain = classmethod(_multinomial_infer_value_domain)  # noqa: F821
Multinomial._infer_value_domain = classmethod(_multinomial_infer_value_domain)  # noqa: F821
DirichletMultinomial._infer_value_domain = classmethod(_multinomial_infer_value_domain)  # noqa: F821


###############################################
# Converting PyTorch Distributions to funsors
###############################################

@to_funsor.register(torch.distributions.Transform)
def transform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    raise NotImplementedError(f"{tfm} is not a currently supported transform")


@to_funsor.register(torch.distributions.transforms.ExpTransform)
def exptransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    name = next(real_inputs.keys()) if real_inputs else "value"
    return ops.exp(Variable(name, output))


@to_funsor.register(torch.distributions.transforms.SigmoidTransform)
def sigmoidtransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    name = next(real_inputs.keys()) if real_inputs else "value"
    return ops.sigmoid(Variable(name, output))


@to_funsor.register(torch.distributions.transforms._InverseTransform)
def inversetransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    return to_funsor(tfm._inv, output=output, dim_to_name=dim_to_name, real_inputs=real_inputs)


@to_funsor.register(torch.distributions.transforms.ComposeTransform)
def composetransform_to_funsor(tfm, output=None, dim_to_name=None, real_inputs=None):
    name = next(real_inputs.keys()) if real_inputs else "value"
    expr = Variable(name, output)
    for part in tfm.parts:  # XXX should this be reversed(parts)?
        expr = to_funsor(part, output=output, dim_to_name=dim_to_name, real_inputs=real_inputs)(**{name: expr})
    return expr


@to_data.register(Unary[ops.TransformOp, Funsor])
def transform_to_data(expr, name_to_dim=None):
    raise NotImplementedError(f"{expr.op} is not a currently supported transform")


@to_data.register(Unary[ops.ExpOp, Variable])
def exptransform_to_data(expr, name_to_dim=None):
    return torch.distributions.transforms.ExpTransform()


@to_data.register(Unary[ops.LogOp, Variable])
def logtransform_to_data(expr, name_to_dim=None):
    return torch.distributions.transforms.ExpTransform().inv


# @to_data.register(Unary[ops.SigmoidOp, Variable])  # TODO create a SigmoidOp class
def sigmoidtransform_to_data(expr, name_to_dim=None):
    return torch.distributions.transforms.SigmoidTransform()


to_funsor.register(torch.distributions.Distribution)(backenddist_to_funsor)
to_funsor.register(torch.distributions.Independent)(indepdist_to_funsor)
to_funsor.register(MaskedDistribution)(maskeddist_to_funsor)
to_funsor.register(torch.distributions.TransformedDistribution)(transformeddist_to_funsor)
to_funsor.register(torch.distributions.MultivariateNormal)(mvndist_to_funsor)


@to_funsor.register(torch.distributions.Bernoulli)
def bernoulli_to_funsor(pyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = _PyroWrapper_BernoulliLogits(logits=pyro_dist.logits)
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


@eager.register(Contraction, ops.LogAddExpOp, ops.AddOp, frozenset, Dirichlet, Multinomial)  # noqa: F821
def eager_dirichlet_multinomial(red_op, bin_op, reduced_vars, x, y):
    dirichlet_reduction = frozenset(x.inputs).intersection(reduced_vars)
    if dirichlet_reduction:
        return DirichletMultinomial(concentration=x.concentration,  # noqa: F821
                                    total_count=y.total_count,
                                    value=y.value)
    else:
        return eager(Contraction, red_op, bin_op, reduced_vars, (x, y))


JointDirichletMultinomial = Contraction[
    Union[ops.LogAddExpOp, ops.NullOp],
    ops.AddOp,
    frozenset,
    Tuple[Dirichlet, Multinomial],  # noqa: F821
]


@eager.register(Binary, ops.SubOp, JointDirichletMultinomial, DirichletMultinomial)  # noqa: F821
def eager_dirichlet_posterior(op, c, z):
    if (z.concentration is c.terms[0].concentration) and (c.terms[1].total_count is z.total_count):
        return Dirichlet(  # noqa: F821
            concentration=z.concentration + c.terms[1].value,
            value=c.terms[0].value)
    else:
        return None
