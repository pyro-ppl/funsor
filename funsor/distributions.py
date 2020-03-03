# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import makefun
import pyro.distributions as dist
import torch

import funsor.delta
import funsor.ops as ops
from funsor.affine import is_affine
from funsor.domains import Domain, bint, reals
from funsor.gaussian import Gaussian
from funsor.interpreter import gensym, interpretation
from funsor.tensor import Tensor, align_tensors, ignore_jit_warnings, stack
from funsor.terms import Funsor, FunsorMeta, Number, Variable, eager, lazy, to_funsor
from funsor.util import broadcast_shape


def _dummy_tensor(domain):
    return torch.tensor(0.1 if domain.dtype == 'real' else 1).expand(domain.shape)


def numbers_to_tensors(*args):
    """
    Convert :class:`~funsor.terms.Number`s to :class:`funsor.tensor.Tensor`s,
    using any provided tensor as a prototype, if available.
    """
    if any(isinstance(x, Number) for x in args):
        options = dict(dtype=torch.get_default_dtype())
        for x in args:
            if isinstance(x, Tensor):
                options = dict(dtype=x.data.dtype, device=x.data.device)
                break
        with ignore_jit_warnings():
            args = tuple(Tensor(torch.tensor(x.data, **options), dtype=x.dtype)
                         if isinstance(x, Number) else x
                         for x in args)
    return args


class DistributionMeta(FunsorMeta):
    """
    Wrapper to fill in default values and convert Numbers to Tensors.
    """
    def __call__(cls, *args, **kwargs):
        kwargs.update(zip(cls._ast_fields, args))
        value = kwargs.pop('value', 'value')
        kwargs = OrderedDict(zip(kwargs.keys(), numbers_to_tensors(*map(to_funsor, kwargs.values()))))
        value = to_funsor(value, output=cls._infer_value_shape(**kwargs))

        args = tuple(kwargs.values()) + (value,)
        return super(DistributionMeta, cls).__call__(*args)


class Distribution(Funsor, metaclass=DistributionMeta):
    r"""
    Funsor backed by a PyTorch distribution object.

    :param \*args: Distribution-dependent parameters.  These can be either
        funsors or objects that can be coerced to funsors via
        :func:`~funsor.terms.to_funsor` . See derived classes for details.
    """
    dist_class = "defined by derived classes"

    def __init__(self, *args):
        params = tuple(zip(self._ast_fields, args))
        assert any(k == 'value' for k, v in params)
        inputs = OrderedDict()
        for name, value in params:
            assert isinstance(name, str)
            assert isinstance(value, Funsor)
            inputs.update(value.inputs)
        inputs = OrderedDict(inputs)
        output = reals()
        super(Distribution, self).__init__(inputs, output)
        self.params = params

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               ', '.join('{}={}'.format(*kv) for kv in self.params))

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp and isinstance(self.value, Variable) and self.value.name in reduced_vars:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).eager_reduce(op, reduced_vars)

    @classmethod
    def eager_log_prob(cls, *params):
        inputs, tensors = align_tensors(*params)
        params = dict(zip(cls._ast_fields, tensors))
        value = params.pop('value')
        data = cls.dist_class(**params).log_prob(value)
        return Tensor(data, inputs)

    def __getattribute__(self, attr):
        if attr in type(self)._ast_fields and attr != 'name':
            return self.params[attr]
        return super().__getattribute__(attr)

    @classmethod
    def _infer_value_shape(cls, **kwargs):
        # rely on the underlying distribution's logic to infer the event_shape
        instance = cls.dist_class(**{k: _dummy_tensor(v.output) for k, v in kwargs.items()}, validate_args=False)
        out_shape = instance.event_shape
        if isinstance(instance.support, torch.distributions.constraints._IntegerInterval):
            out_dtype = instance.support.upper_bound + 1
        else:
            out_dtype = 'real'
        return Domain(dtype=out_dtype, shape=out_shape)


################################################################################
# Distribution Wrappers
################################################################################

def make_dist(pyro_dist_class, param_names=()):

    if not param_names:
        param_names = tuple(pyro_dist_class.arg_constraints.keys())
    assert all(name in pyro_dist_class.arg_constraints for name in param_names)

    @makefun.with_signature(f"__init__(self, {', '.join(param_names)}, value='value')")
    def dist_init(self, *args, **kwargs):
        return Distribution.__init__(self, *tuple(kwargs.values()))

    dist_class = DistributionMeta(pyro_dist_class.__name__, (Distribution,), {
        'dist_class': pyro_dist_class,
        '__init__': dist_init,
    })

    eager.register(dist_class, *((Tensor,) * (len(param_names) + 1)))(dist_class.eager_log_prob)

    return dist_class


class BernoulliProbs(dist.Bernoulli):
    def __init__(self, probs, validate_args=None):
        return super().__init__(probs=probs, validate_args=validate_args)


class BernoulliLogits(dist.Bernoulli):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


class CategoricalLogits(dist.Categorical):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


_wrapped_pyro_dists = [
    (dist.Beta, ()),
    (BernoulliProbs, ('probs',)),
    (BernoulliLogits, ('logits',)),
    (dist.Binomial, ('total_count', 'probs')),
    # (dist.Multinomial, ('total_count', 'probs')),  # TODO
    (dist.Categorical, ('probs',)),
    (CategoricalLogits, ('logits',)),
    (dist.Poisson, ()),
    (dist.Gamma, ()),
    (dist.VonMises, ()),
    (dist.Dirichlet, ()),
    # (dist.DirichletMultinomial, ()),
    (dist.Normal, ()),
    (dist.MultivariateNormal, ('loc', 'scale_tril')),
    (dist.Delta, ()),
]

for pyro_dist_class, param_names in _wrapped_pyro_dists:
    locals()[pyro_dist_class.__name__.split(".")[-1]] = make_dist(pyro_dist_class, param_names)


####################

def Bernoulli(probs=None, logits=None, value='value'):
    """
    Wraps :class:`pyro.distributions.Bernoulli` .

    This dispatches to either :class:`BernoulliProbs` or
    :class:`BernoulliLogits` to accept either ``probs`` or ``logits`` args.

    :param Funsor probs: Probability of 1.
    :param Funsor value: Optional observation in ``{0,1}``.
    """
    if probs is not None:
        return BernoulliProbs(probs, value)
    if logits is not None:
        return BernoulliLogits(logits, value)
    raise ValueError('Either probs or logits must be specified')


@eager.register(Beta, Funsor, Funsor, Funsor)
def eager_beta(concentration1, concentration0, value):
    concentration = stack((concentration0, concentration1))
    value = stack((1 - value, value))
    return Dirichlet(concentration, value=value)


@eager.register(Binomial, Funsor, Funsor, Funsor)
def eager_binomial(total_count, probs, value):
    probs = stack((1 - probs, probs))
    value = stack((total_count - value, value))
    return Multinomial(total_count, probs, value=value)


@eager.register(Categorical, Funsor, Tensor)
def eager_categorical(probs, value):
    return probs[value].log()


@eager.register(Categorical, Tensor, Variable)
def eager_categorical(probs, value):
    value = probs.materialize(value)
    return Categorical(probs=probs, value=value)


@eager.register(Delta, Tensor, Tensor, Tensor)
def eager_delta(v, log_density, value):
    # This handles event_dim specially, and hence cannot use the
    # generic Delta.eager_log_prob() method.
    assert v.output == value.output
    event_dim = len(v.output.shape)
    inputs, (v, log_density, value) = align_tensors(v, log_density, value)
    data = dist.Delta(v, log_density, event_dim).log_prob(value)
    return Tensor(data, inputs)


@eager.register(Delta, Funsor, Funsor, Variable)
@eager.register(Delta, Variable, Funsor, Variable)
def eager_delta(v, log_density, value):
    assert v.output == value.output
    return funsor.delta.Delta(value.name, v, log_density)


@eager.register(Delta, Variable, Funsor, Funsor)
def eager_delta(v, log_density, value):
    assert v.output == value.output
    return funsor.delta.Delta(v.name, value, log_density)


def LogNormal(loc, scale, value='value'):
    """
    Wraps :class:`pyro.distributions.LogNormal` .

    :param Funsor loc: Mean of the untransformed Normal distribution.
    :param Funsor scale: Standard deviation of the untransformed Normal
        distribution.
    :param Funsor value: Optional real observation.
    """
    loc, scale, y = Normal._fill_defaults(loc, scale, value)
    t = ops.exp
    x = t.inv(y)
    log_abs_det_jacobian = t.log_abs_det_jacobian(x, y)
    return Normal(loc, scale, x) - log_abs_det_jacobian


@eager.register(Normal, Funsor, Tensor, Funsor)
def eager_normal(loc, scale, value):
    assert loc.output == reals()
    assert scale.output == reals()
    assert value.output == reals()
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


@eager.register(MultivariateNormal, Funsor, Tensor, Funsor)
def eager_mvn(loc, scale_tril, value):
    assert len(loc.shape) == 1
    assert len(scale_tril.shape) == 2
    assert value.output == loc.output
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


# __all__ = [
#     'Bernoulli',
#     'BernoulliLogits',
#     'Beta',
#     'Binomial',
#     'Categorical',
#     'Delta',
#     'Dirichlet',
#     'DirichletMultinomial',
#     'Distribution',
#     'Gamma',
#     'LogNormal',
#     'Multinomial',
#     'MultivariateNormal',
#     'Normal',
#     'Poisson',
#     'VonMises',
# ]
