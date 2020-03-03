# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict
from typing import Tuple

import makefun
import torch
import pyro.distributions as dist
from pyro.distributions.torch_distribution import MaskedDistribution
from pyro.distributions.util import broadcast_shape

import funsor.ops as ops
from funsor.affine import is_affine
from funsor.domains import Domain, bint, reals
from funsor.gaussian import Gaussian
from funsor.interpreter import gensym
from funsor.registry import KeyedRegistry
from funsor.tensor import Tensor, align_tensors, stack
from funsor.terms import Funsor, FunsorMeta, Independent, Number, Subs, Variable, eager, to_data, to_funsor


def _dummy_tensor(domain):
    return torch.tensor(0.1 if domain.dtype == 'real' else 1).expand(domain.shape)


class DistributionMeta2(FunsorMeta):
    def __call__(cls, *args, name=None):
        if len(args) < len(cls._ast_fields):
            args = args + (name if name is not None else 'value',)
        return super(DistributionMeta2, cls).__call__(*args)


class Distribution2(Funsor, metaclass=DistributionMeta2):
    """
    Different design for the Distribution Funsor wrapper,
    closer to Gaussian or Delta in which the value is a fresh input.
    """
    dist_class = dist.Distribution  # defined by derived classes

    def __init__(self, *args, name='value'):
        params = OrderedDict(zip(self._ast_fields, args))
        inputs = OrderedDict()
        for param_name, value in params.items():
            assert isinstance(param_name, str)
            assert isinstance(value, Funsor)
            inputs.update(value.inputs)
        assert isinstance(name, str) and name not in inputs
        inputs[name] = self._infer_value_shape(**params)
        output = reals()
        fresh = frozenset({name})
        bound = frozenset()
        super().__init__(inputs, output, fresh, bound)
        self.params = params
        self.name = name

    def __getattribute__(self, attr):
        if attr in type(self)._ast_fields and attr != 'name':
            return self.params[attr]
        return super().__getattribute__(attr)

    @classmethod
    def _infer_value_shape(cls, **kwargs):
        # rely on the underlying distribution's logic to infer the event_shape
        instance = cls.dist_class(**{k: _dummy_tensor(v.output) for k, v in kwargs.items()})
        out_shape = instance.event_shape
        if isinstance(instance.support, torch.distributions.constraints._IntegerInterval):
            out_dtype = instance.support.upper_bound + 1
        else:
            out_dtype = 'real'
        return Domain(dtype=out_dtype, shape=out_shape)

    def eager_subs(self, subs):
        name, sub = subs[0]
        args = tuple(self.params.values()) + (sub,)
        result = dist_eager_subs(type(self), *args)
        if result is None:
            if all(isinstance(v, (Number, Tensor)) for v in args):
                inputs, tensors = align_tensors(*args)
                data = self.dist_class(*tensors[:-1]).log_prob(tensors[-1])
                result = Tensor(data, inputs)
            elif isinstance(sub, Variable):
                result = type(self)(*self._ast_values, name=sub.name)
        return result


######################################
# Converting distributions to funsors
######################################

@to_funsor.register(torch.distributions.Distribution)
def torchdistribution_to_funsor(pyro_dist, output=None, dim_to_name=None):
    import funsor.distributions2  # TODO find a better way to do this lookup
    funsor_dist_class = getattr(funsor.distributions2, type(pyro_dist).__name__)
    params = [to_funsor(getattr(pyro_dist, param_name), dim_to_name=dim_to_name)
              for param_name in funsor_dist_class._ast_fields if param_name != 'name']
    return funsor_dist_class(*params)


@to_funsor.register(torch.distributions.Independent)
def indepdist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    result = to_funsor(pyro_dist.base_dist, dim_to_name=dim_to_name)
    for i in range(pyro_dist.reinterpreted_batch_ndims):
        name = ...  # XXX what is this? read off from result?
        result = funsor.terms.Independent(result, "value", name, "value")
    return result


@to_funsor.register(MaskedDistribution)
def maskeddist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    mask = to_funsor(pyro_dist._mask.float(), output=output, dim_to_name=dim_to_name)
    funsor_base_dist = to_funsor(pyro_dist.base_dist, output=output, dim_to_name=dim_to_name)
    return mask * funsor_base_dist


@to_funsor.register(torch.distributions.TransformedDistribution)
def transformeddist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    raise NotImplementedError("TODO")


# @to_funsor.register(FunsorDistribution)
# def funsordist_to_funsor(pyro_dist, output=None, dim_to_name=None):
#     rename = ...  # TODO use dim_to_name to rename batch dims
#     return pyro_dist.value(**rename)


###########################################################
# Converting distribution funsors to PyTorch distributions
###########################################################

@to_data.register(Distribution2)
def distribution_to_data(funsor_dist, name_to_dim=None):
    pyro_dist_class = funsor_dist.dist_class
    params = [to_data(getattr(funsor_dist, param_name), name_to_dim=name_to_dim)
              for param_name in funsor_dist._ast_fields if param_name != 'name']
    pyro_dist = pyro_dist_class(*params)
    funsor_event_shape = funsor_dist.inputs[funsor_dist.name].shape
    pyro_dist = pyro_dist.to_event(max(len(funsor_event_shape) - len(pyro_dist.event_shape), 0))
    if pyro_dist.event_shape != funsor_event_shape:
        raise ValueError("Event shapes don't match, something went wrong")
    return pyro_dist


@to_data.register(Independent)
def indep_to_data(funsor_dist, name_to_dim=None):
    return to_data(funsor_dist.term, name_to_dim).to_event(len(funsor_dist.inputs[funsor_dist.name].shape))


################################################################################
# Distribution Wrappers
################################################################################

def make_dist(pyro_dist_class, param_names=()):

    if not param_names:
        param_names = tuple(pyro_dist_class.arg_constraints.keys())
    assert all(name in pyro_dist_class.arg_constraints for name in param_names)

    @makefun.with_signature(f"__init__(self, {', '.join(param_names)}, name='value')")
    def dist_init(self, *args, **kwargs):
        return Distribution2.__init__(self, *tuple(kwargs.values())[:-1], name=kwargs['name'])

    dist_class = DistributionMeta2(pyro_dist_class.__name__, (Distribution2,), {
        'dist_class': pyro_dist_class,
        '__init__': dist_init,
    })

    eager.register(dist_class, *((Tensor,) * len(param_names) + 1))(dist_class.eager_log_prob)

    return dist_class


class BernoulliProbs(dist.Bernoulli):
    def __init__(self, probs, validate_args=None):
        return super().__init__(probs=probs, validate_args=validate_args)


class BernoulliLogits(dist.Bernoulli):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


class CategoricalProbs(dist.Categorical):
    def __init__(self, probs, validate_args=None):
        return super().__init__(probs=probs, validate_args=validate_args)


class CategoricalLogits(dist.Categorical):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


_wrapped_pyro_dists = [
    (dist.Beta, ()),
    (BernoulliProbs, ('probs',)),
    (BernoulliLogits, ('logits',)),
    (dist.Binomial, ('total_count', 'probs')),
    # (dist.Multinomial, ('total_count', 'probs')),  # TODO
    (CategoricalProbs, ('probs',)),
    (CategoricalLogits, ('logits',)),
    (dist.Poisson, ()),
    (dist.Gamma, ()),
    (dist.VonMises, ()),
    (dist.Dirichlet, ()),
    (dist.Normal, ()),
    (dist.MultivariateNormal, ('loc', 'scale_tril')),
]

for pyro_dist_class, param_names in _wrapped_pyro_dists:
    locals()[pyro_dist_class.__name__.split(".")[-1]] = make_dist(pyro_dist_class, param_names)


###########################
# Distribution patterns
###########################

dist_eager_subs = KeyedRegistry(default=lambda *args: None)


@dist_eager_subs.register(Beta, Funsor, Funsor, Funsor)
def eager_beta(concentration1, concentration0, value):
    concentration = stack((concentration0, concentration1))
    value = stack((1 - value, value))
    return Dirichlet(concentration)(value=value)


@dist_eager_subs.register(Binomial, Funsor, Funsor, Funsor)
def eager_binomial(total_count, probs, value):
    probs = stack((1 - probs, probs))
    value = stack((total_count - value, value))
    return Multinomial(total_count, probs)(value=value)


@dist_eager_subs.register(CategoricalProbs, Funsor, Tensor)
def eager_categorical(probs, value):
    return probs[value].log()


@dist_eager_subs.register(CategoricalProbs, Tensor, Variable)
def eager_categorical(probs, value):
    return CategoricalProbs(probs)(value=probs.materialize(value))


@dist_eager_subs.register(Normal, Funsor, Tensor, Funsor)
def eager_normal(loc, scale, value):
    if not is_affine(loc) or not is_affine(value):
        return None  # lazy

    info_vec = scale.data.new_zeros(scale.data.shape + (1,))
    precision = scale.data.pow(-2).reshape(scale.data.shape + (1, 1))
    log_prob = -0.5 * math.log(2 * math.pi) - scale.log().sum()
    inputs = scale.inputs.copy()
    var = gensym('value')
    inputs[var] = reals()
    gaussian = log_prob + Gaussian(info_vec, precision, inputs)
    return gaussian(**{var: value - loc})


@dist_eager_subs.register(MultivariateNormal, Funsor, Tensor, Funsor)
def eager_mvn(loc, scale_tril, value):
    if not is_affine(loc) or not is_affine(value):
        return None  # lazy

    info_vec = scale_tril.data.new_zeros(scale_tril.data.shape[:-1])
    precision = ops.cholesky_inverse(scale_tril.data)
    scale_diag = Tensor(scale_tril.data.diagonal(dim1=-1, dim2=-2), scale_tril.inputs)
    log_prob = -0.5 * scale_diag.shape[0] * math.log(2 * math.pi) - scale_diag.log().sum()
    inputs = scale_tril.inputs.copy()
    var = gensym('value')
    inputs[var] = reals(scale_diag.shape[0])
    gaussian = log_prob + Gaussian(info_vec, precision, inputs)
    return gaussian(**{var: value - loc})
