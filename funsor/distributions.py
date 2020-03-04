# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import makefun
import pyro.distributions as dist
from pyro.distributions.torch_distribution import MaskedDistribution
import torch
import torch.distributions.constraints as constraints

import funsor.delta
import funsor.ops as ops
from funsor.affine import is_affine
from funsor.domains import Domain, reals
from funsor.gaussian import Gaussian
from funsor.interpreter import gensym
from funsor.tensor import Tensor, align_tensors, ignore_jit_warnings, stack
from funsor.terms import Funsor, FunsorMeta, Independent, Number, Variable, eager, to_data, to_funsor
from funsor.util import broadcast_shape


def _dummy_tensor(domain):
    value = 0.1 if domain.dtype == 'real' else 1
    return torch.tensor(value).expand(domain.shape) if domain.shape else value


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
        kwargs = OrderedDict((k, to_funsor(v, output=cls._infer_param_domain(k, v)))
                             for k, v in kwargs.items())
        value = to_funsor(value, output=cls._infer_value_domain(**kwargs))
        args = numbers_to_tensors(*(tuple(kwargs.values()) + (value,)))
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
    def _infer_value_domain(cls, **kwargs):
        # rely on the underlying distribution's logic to infer the event_shape
        instance = cls.dist_class(**{k: _dummy_tensor(v.output) for k, v in kwargs.items()}, validate_args=False)
        out_shape = instance.event_shape
        if isinstance(instance.support, constraints._IntegerInterval):
            out_dtype = int(instance.support.upper_bound + 1)
            # this is a hack, but we don't really care about precise dtypes except for Categorical
            out_dtype = 'real' if out_dtype == 1 else out_dtype
        else:
            out_dtype = 'real'
        return Domain(dtype=out_dtype, shape=out_shape)

    @classmethod
    def _infer_param_domain(cls, name, raw_value):
        support = cls.dist_class.arg_constraints.get(name, None)
        if isinstance(support, constraints._Simplex):
            output = reals(raw_value.shape[-1])
        elif isinstance(support, constraints._RealVector):
            output = reals(raw_value.shape[-1])
        elif isinstance(support, (constraints._LowerCholesky, constraints._PositiveDefinite)):
            output = reals(*raw_value.shape[-2:])
        elif isinstance(support, constraints._Real) and name == "logits" and \
                isinstance(cls.dist_class.arg_constraints["probs"], constraints._Simplex):
            output = reals(raw_value.shape[-1])
        else:
            output = None
        return output


################################################################################
# Distribution Wrappers
################################################################################

def make_dist(pyro_dist_class, param_names=()):

    if not param_names:
        param_names = tuple(pyro_dist_class.arg_constraints.keys())

    @makefun.with_signature(f"__init__(self, {', '.join(param_names)}, value='value')")
    def dist_init(self, *args, **kwargs):
        return Distribution.__init__(self, *tuple(kwargs.values()))

    dist_class = DistributionMeta(pyro_dist_class.__name__.split("__")[-1], (Distribution,), {
        'dist_class': pyro_dist_class,
        '__init__': dist_init,
    })

    eager.register(dist_class, *((Tensor,) * (len(param_names) + 1)))(dist_class.eager_log_prob)

    return dist_class


class __BernoulliProbs(dist.Bernoulli):
    def __init__(self, probs, validate_args=None):
        return super().__init__(probs=probs, validate_args=validate_args)


class __BernoulliLogits(dist.Bernoulli):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


class __CategoricalLogits(dist.Categorical):
    def __init__(self, logits, validate_args=None):
        return super().__init__(logits=logits, validate_args=validate_args)


_wrapped_pyro_dists = [
    (dist.Beta, ()),
    (__BernoulliProbs, ('probs',)),
    (__BernoulliLogits, ('logits',)),
    (dist.Binomial, ('total_count', 'probs')),
    (dist.Multinomial, ('total_count', 'probs')),
    (dist.Categorical, ('probs',)),
    (__CategoricalLogits, ('logits',)),
    (dist.Poisson, ()),
    (dist.Gamma, ()),
    (dist.VonMises, ()),
    (dist.Dirichlet, ()),
    (dist.DirichletMultinomial, ()),
    (dist.Normal, ()),
    (dist.MultivariateNormal, ('loc', 'scale_tril')),
    (dist.Delta, ()),
]

for pyro_dist_class, param_names in _wrapped_pyro_dists:
    locals()[pyro_dist_class.__name__.split("__")[-1].split(".")[-1]] = make_dist(pyro_dist_class, param_names)

# Delta has to be treated specially because of its weird shape inference semantics
Delta._infer_value_domain = classmethod(lambda cls, **kwargs: kwargs['v'].output)


###############################################
# Converting PyTorch Distributions to funsors
###############################################

@to_funsor.register(torch.distributions.Distribution)
def torchdistribution_to_funsor(pyro_dist, output=None, dim_to_name=None):
    import funsor.distributions  # TODO find a better way to do this lookup
    funsor_dist_class = getattr(funsor.distributions, type(pyro_dist).__name__.strip("_"))
    params = [to_funsor(getattr(pyro_dist, param_name),
                        output=funsor_dist_class._infer_param_domain(param_name, getattr(pyro_dist, param_name)),
                        dim_to_name=dim_to_name)
              for param_name in funsor_dist_class._ast_fields if param_name != 'value']
    return funsor_dist_class(*params)


@to_funsor.register(torch.distributions.Independent)
def indepdist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    result = to_funsor(pyro_dist.base_dist, dim_to_name=dim_to_name)
    for i in range(pyro_dist.reinterpreted_batch_ndims):
        name = f"dim_{i}"  # XXX what is this? read off from dim_to_name? does it matter?
        result = funsor.terms.Independent(result, "value", name, "value")
    return result


@to_funsor.register(MaskedDistribution)
def maskeddist_to_funsor(pyro_dist, output=None, dim_to_name=None):
    mask = to_funsor(pyro_dist._mask.float(), output=output, dim_to_name=dim_to_name)
    funsor_base_dist = to_funsor(pyro_dist.base_dist, output=output, dim_to_name=dim_to_name)
    return mask * funsor_base_dist


@to_funsor.register(dist.Bernoulli)
def bernoulli_to_funsor(pyro_dist, output=None, dim_to_name=None):
    new_pyro_dist = __BernoulliLogits(logits=pyro_dist.logits)
    return torchdistribution_to_funsor(new_pyro_dist, output, dim_to_name)


###########################################################
# Converting distribution funsors to PyTorch distributions
###########################################################

@to_data.register(Distribution)
def distribution_to_data(funsor_dist, name_to_dim=None):
    pyro_dist_class = funsor_dist.dist_class
    params = [to_data(getattr(funsor_dist, param_name), name_to_dim=name_to_dim)
              for param_name in funsor_dist._ast_fields if param_name != 'value']
    pyro_dist = pyro_dist_class(*params)
    funsor_event_shape = funsor_dist.value.output.shape
    pyro_dist = pyro_dist.to_event(max(len(funsor_event_shape) - len(pyro_dist.event_shape), 0))
    if pyro_dist.event_shape != funsor_event_shape:
        raise ValueError("Event shapes don't match, something went wrong")
    return pyro_dist


@to_data.register(Independent)
def indep_to_data(funsor_dist, name_to_dim=None):
    return to_data(funsor_dist.term, name_to_dim).to_event(len(funsor_dist.value.output.shape))


################################################
# Backend-agnostic distribution patterns
################################################

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


@eager.register(Multinomial, Tensor, Tensor, Tensor)
def eager_multinomial(total_count, probs, value):
    # Multinomial.log_prob() supports inhomogeneous total_count only by
    # avoiding passing total_count to the constructor.
    inputs, (total_count, probs, value) = align_tensors(total_count, probs, value)
    shape = broadcast_shape(total_count.shape + (1,), probs.shape, value.shape)
    probs = Tensor(probs.expand(shape), inputs)
    value = Tensor(value.expand(shape), inputs)
    total_count = Number(total_count.max().item())  # Used by distributions validation code.
    return Multinomial.eager_log_prob(total_count, probs, value)


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
    loc, scale = to_funsor(loc), to_funsor(scale)
    y = to_funsor(value, output=loc.output)
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
