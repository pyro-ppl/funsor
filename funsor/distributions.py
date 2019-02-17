from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import torch.distributions as dist
from multipledispatch import dispatch
from six import add_metaclass

import funsor.ops as ops
from funsor.adjoint import backward
from funsor.contract import contract
from funsor.pattern import simplify_sum
from funsor.six import getargspec
from funsor.terms import Funsor, FunsorMeta, Number, Variable, to_funsor
from funsor.torch import Tensor, align_tensors


def log_abs_det(jacobian):
    result = 0.
    for i, row in jacobian.items():
        for j, entry in row.items():
            if i != j:
                raise NotImplementedError('TODO handle non-diagonal jacobians')
                result += ops.log(entry)
    return result


def log_abs_det_jacobian(transform):
    jacobian = defaultdict(dict)
    for key, value in transform.items():
        for dim in value.dims:
            jacobian[key][dim] = value.jacobian(dim)
    return log_abs_det(jacobian)


def match_affine(expr, dim):
    assert isinstance(expr, Funsor)
    assert isinstance(dim, str)
    a1 = expr.jacobian(dim)
    if dim not in a1.dims:
        a0 = expr(**{dim: 0.})
        yield a0, a1


# WIP candidate base distribution interface
class AbstractDistribution(object):
    def __init__(self, samples, log_prob):
        assert isinstance(samples, dict)
        assert isinstance(log_prob, Funsor)
        for k, v in samples.items():
            assert isinstance(k, str)
            assert isinstance(v, Funsor)
            assert set(v.dims) == set(log_prob.dims)
        super(AbstractDistribution, self).__init__()
        self.samples = samples
        self.log_prob = log_prob

    @property
    def dims(self):
        return self.log_prob.dims

    def sample(self, dims):
        if not dims:
            return self
        log_prob = self.log_prob.reduce(ops.sample, dims)
        samples = backward(ops.sample, log_prob)
        samples.update((k, v(**samples)) for k, v in self.samples)
        return AbstractDistribution(samples, log_prob)


DEFAULT_VALUE = Variable('value', 'real')


class DefaultValueMeta(FunsorMeta):
    def __call__(cls, *args, **kwargs):
        # TODO do this once on class init.
        if not hasattr(cls, '_ast_fields'):
            cls._ast_fields = getargspec(cls.__init__)[0][1:]

        kwargs.update(zip(cls._ast_fields, args))
        kwargs.setdefault('value', DEFAULT_VALUE)
        return super(DefaultValueMeta, cls).__call__(**kwargs)


@add_metaclass(DefaultValueMeta)
class Distribution(Funsor):
    """
    Abstract base class for funsors representing univariate probability
    distributions.
    """
    def __init__(self, cls, **params):
        assert issubclass(cls, dist.Distribution)
        schema = params['value'].schema.copy()
        self.params = OrderedDict()
        for k, v in sorted(params.items()):
            assert isinstance(k, str)
            v = to_funsor(v)
            schema.update(v.schema)
            self.params[k] = v
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(Distribution, self).__init__(dims, shape)
        self.cls = cls

    @property
    def value(self):
        return self.params['value']

    @property
    def is_observed(self):
        return isinstance(self.value, (Number, Tensor))

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               ', '.join('{}={}'.format(*kv) for kv in self.params.items()))

    def _eager_subs(self, **kwargs):
        if not kwargs:
            return self
        params = OrderedDict((k, v(**kwargs)) for k, v in self.params.items())
        if all(isinstance(v, (Number, Tensor)) for v in params.values()):
            dims, tensors = align_tensors(*params.values())
            params = dict(zip(params, tensors))
            value = params.pop('value')
            data = self.cls(**params).log_prob(value)
            return Tensor(dims, data)
        return type(self)(**params)

    def reduce(self, op, dims):
        if op is ops.logaddexp and isinstance(self.value, Variable) and self.value.name in dims:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).reduce(op, dims)

    # Legacy distributions interface:

    def log_prob(self, value, event_dims=()):
        assert isinstance(self.value, Variable)
        # TODO handle event dims
        return self(**{self.value.name: value})

    def sample(self):
        return backward(ops.sample, self, frozenset(self.value.dims))

    def transform(self, **transform):
        return self(**transform) + log_abs_det_jacobian(transform)  # sign error?


@backward.register(ops.sample, Distribution)
def _sample_torch_distribution(term, dims):
    if len(dims) != 1:
        raise NotImplementedError('TODO')
    value = term.value
    if not isinstance(value, Variable):
        raise NotImplementedError
    if value.name not in dims:
        raise NotImplementedError('TODO')

    params = term.params.copy()
    params.pop('value')
    if all(isinstance(v, (Number, Tensor)) for v in params.values()):
        dims, tensors = align_tensors(*params.values())
        params = dict(zip(params, tensors))
        data = term.cls(**params).rsample()
        return {value.name: Tensor(dims, data)}

    raise NotImplementedError('TODO')


################################################################################
# Distribution Wrappers
################################################################################


class Beta(Distribution):
    def __init__(self, concentration1, concentration0, value=DEFAULT_VALUE):
        super(Beta, self).__init__(dist.Beta, value=value,
                                   concentration1=concentration1, concentration0=concentration0)


class Gamma(Distribution):
    def __init__(self, concentration, rate, value=DEFAULT_VALUE):
        super(Gamma, self).__init__(dist.Gamma, value=value,
                                    concentration=concentration, rate=rate)


class Binomial(Distribution):
    def __init__(self, total_count, probs, value=DEFAULT_VALUE):
        super(Binomial, self).__init__(dist.Binomial, value=value,
                                       total_count=total_count, probs=probs)


class Delta(Distribution):
    def __init__(self, v, log_density, value=DEFAULT_VALUE):
        super(Delta, self).__init__(dist.Delta, value=value, v=v, log_density=log_density)


class Normal(Distribution):
    def __init__(self, loc, scale, value=DEFAULT_VALUE):
        super(Normal, self).__init__(dist.Normal, value=value, loc=loc, scale=scale)


################################################################################
# Conjugacy Relationships
################################################################################

def _binary_normal_normal(lhs, rhs):
    d = lhs.value.name
    if d not in rhs.params['scale'].dims and d in rhs.params['loc'].dims:
        for a0, a1 in match_affine(rhs.params['loc'], d):
            print('UPDATE\n lhs = {}\n rhs = {}'.format(lhs, rhs))
            loc1, scale1 = lhs.params['loc'], lhs.params['scale']
            scale2 = rhs.params['scale']
            loc2 = (rhs.value - a0) / a1
            scale2 = scale2 / a1.abs()
            prec1 = scale1 ** -2
            prec2 = scale2 ** -2

            # Perform a filter update.
            prec3 = prec1 + prec2
            loc3 = loc1 + simplify_sum(loc2 - loc1) * (prec2 / prec3)
            scale3 = prec3 ** -0.5
            updated = Normal(loc3, scale3, value=lhs.value)
            # FIXME add log(a1) term?

            # Encapsulate the log_likelihood in a Normal for later pattern matching.
            prec4 = prec1 * prec2 / prec3
            scale4 = prec4 ** -0.5
            loc4 = simplify_sum(loc2 - loc1)
            log_likelihood = Normal(loc4, scale4)(value=0.) + a1.abs().log()  # FIXME
            result = updated + log_likelihood
            print(' result = {}'.format(result))
            return result


@dispatch(object, Normal, Normal)
def eager_binary(op, lhs, rhs):
    # Try updating prior from a ground observation.
    if op is ops.add:
        if isinstance(lhs.value, Variable) and rhs.is_observed:
            return _binary_normal_normal(lhs, rhs)
        if isinstance(rhs.value, Variable) and lhs.is_observed:
            return _binary_normal_normal(lhs, rhs)
        raise NotImplementedError('TODO')


@contract.register((ops.logaddexp, ops.mul), Delta, Funsor)
@contract.register((ops.logaddexp, ops.mul), Delta, Delta)
def _contract_delta(lhs, rhs):
    return rhs(value=lhs.params['v'])


@contract.register((ops.logaddexp, ops.mul), Normal, Normal)
def _contract_normal_normal(lhs, rhs, reduce_dims):
    raise NotImplementedError('TODO')


__all__ = [
    'Beta',
    'Binomial',
    'Delta',
    'Distribution',
    'Gamma',
    'Normal',
]
