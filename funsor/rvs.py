from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
from collections import OrderedDict
from six import add_metaclass

import funsor.distributions
import funsor.ops as ops
from funsor.distributions import Normal, numbers_to_tensors
from funsor.domains import reals
from funsor.ops import AssociativeOp, TransformOp
from funsor.optimizer import optimize
from funsor.terms import Binary, Funsor, FunsorMeta, Number, Reduce, Subs, Unary, Variable, \
    eager, reflect, to_funsor
from funsor.torch import Tensor, align_tensors


#########################################################
# Random variables built on PyTorch distributions
#########################################################

PDIST_TO_FDIST = {
    dist.Normal: funsor.distributions.Normal,
    dist.Categorical: funsor.distributions.Categorical,
    dist.Bernoulli: funsor.distributions.Bernoulli,
    dist.Delta: funsor.distributions.Delta,
}


class RandomVariableMeta(FunsorMeta):
    """
    Wrapper to fill in default values and convert Numbers to Tensors.
    """
    def __call__(cls, *args, **kwargs):
        # TODO move omega-handling logic and to_funsor-ing logic here?
        kwargs.update(zip(cls._ast_fields, args))
        args = cls._fill_defaults(**kwargs)  # TODO get rid of _fill_defaults?
        args = numbers_to_tensors(*args)
        return super(RandomVariableMeta, cls).__call__(*args)


@add_metaclass(RandomVariableMeta)
class RandomVariable(Funsor):
    """
    Random Variable class backed by a PyTorch distribution.
    Transforms like a sample, but carries a deferred random number generator call.
    """
    dist_class = "defined by derived classes"

    def __init__(self, *args):
        params = tuple(zip(self._ast_fields, args))
        assert any(k == 'omega' for k, v in params)
        inputs = OrderedDict()
        for name, value in params:
            assert isinstance(name, str)
            assert isinstance(value, Funsor)
            inputs.update(value.inputs)
        inputs = OrderedDict(inputs)
        output = reals()
        super(RandomVariable, self).__init__(inputs, output)
        self.params = OrderedDict(params)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               ', '.join('{}={}'.format(*kv) for kv in self.params))

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        if not any(k in self.inputs for k, v in subs):
            return self
        if any(k == self.params["omega"].name and not isinstance(v, Variable) for k, v in subs):
            raise ValueError("Cannot substitute ground term into omega")
        params = OrderedDict((k, Subs(v, subs)) for k, v in self.params.items())
        return type(self)(**params)

    @property
    def omega(self):
        return self.params["omega"]

    @classmethod
    def eager_dist(cls, **params):
        params.pop('omega', None)
        inputs, tensors = align_tensors(*params.values())
        params = dict(zip(params, tensors))
        return cls.dist_class(**params), inputs

    def eager_reduce(self, op, reduced_vars):
        if op is ops.add and self.omega.name in reduced_vars:
            # expected value...
            # TODO allow propagation of free discrete vars?
            reduced_vars = reduced_vars - {self.omega.name}
            new_params = {k: v.eager_reduce(op, reduced_vars) if reduced_vars else v
                          for k, v in self.params.items() if k != 'omega'}
            # XXX not always correct?
            ground_dist, inputs = self.eager_dist(**new_params)
            return Tensor(ground_dist.mean, inputs)
        return super(RandomVariable, self).eager_reduce(op, reduced_vars)

    def unscaled_sample(self, sampled_vars, sample_inputs):
        if self.omega.name in sampled_vars:
            sampled_vars = sampled_vars - {self.omega.name}
            # FIXME this is not correct, might not share sample values
            # just want to sample parameters here...
            new_params = {k: v.sample(sampled_vars, sample_inputs) if sampled_vars else v
                          for k, v in self.params.items() if k != 'omega'}
            # TODO push a sample_shape built from sample_inputs...
            ground_dist, inputs = self.eager_dist(**new_params)
            return Tensor(ground_dist.sample(), inputs)
        return super(RandomVariable, self).unscaled_sample(sampled_vars, sample_inputs)

    def log_prob(self, value):
        return PDIST_TO_FDIST[self.dist_class](
            **{k: v for k, v in self.params.items() if k != 'omega' and k != 'value'}
        )(value=value)


class CategoricalRV(RandomVariable):
    dist_class = dist.Categorical

    @staticmethod
    def _fill_defaults(probs, omega='omega'):
        probs = to_funsor(probs)
        omega = to_funsor(omega, reals())
        return probs, omega

    def __init__(self, probs, omega='omega'):
        return super(CategoricalRV, self).__init__(probs, omega)


class BernoulliRV(RandomVariable):
    dist_class = dist.Bernoulli

    @staticmethod
    def _fill_defaults(probs, omega='omega'):
        probs = to_funsor(probs)
        omega = to_funsor(omega, reals())
        return probs, omega

    def __init__(self, probs, omega='omega'):
        return super(BernoulliRV, self).__init__(probs, omega)


class NormalRV(RandomVariable):
    dist_class = dist.Normal

    @staticmethod
    def _fill_defaults(loc, scale, omega='omega'):
        loc = to_funsor(loc)
        scale = to_funsor(scale)
        assert loc.output == reals()
        assert scale.output == reals()
        omega = to_funsor(omega, reals())
        return loc, scale, omega

    def __init__(self, loc, scale, omega='omega'):
        return super(NormalRV, self).__init__(loc, scale, omega)

    @property
    def loc(self):
        return self.params['loc']

    @property
    def scale(self):
        return self.params['scale']


class DeltaRV(RandomVariable):
    dist_class = dist.Delta

    @staticmethod
    def _fill_defaults(v, log_density=0, omega='omega'):
        v = to_funsor(v)
        log_density = to_funsor(log_density)
        omega = to_funsor(omega, reals())
        return v, log_density, omega

    def __init__(self, v, log_density, omega='omega'):
        return super(DeltaRV, self).__init__(v, log_density, omega)


##########################################################
# Handling reductions of expressions with RandomVariables
##########################################################

@optimize.register(Normal, RandomVariable, (Funsor, Tensor), (Funsor, Tensor))
@eager.register(Normal, RandomVariable, (Funsor, Tensor), (Funsor, Tensor))
def eager_normaldist_rv(loc, scale, value):
    return reflect(Subs, Normal('locvar', scale, value), (('locvar', loc),))


@optimize.register(Reduce, AssociativeOp, Subs, frozenset)
@eager.register(Reduce, AssociativeOp, Subs, frozenset)
def eager_reduce_subs_rv(op, arg, reduced_vars):
    subs = tuple(arg.subs.items())
    assert not any(k in reduced_vars for (k, v) in subs)
    if op is not ops.logaddexp:
        raise NotImplementedError("other reductions with RV exprs not implemented")

    if all(isinstance(v, RandomVariable) for (k, v) in subs):  # TODO handle partial
        log_density = sum(v.log_prob(k) for (k, v) in subs if v.omega.name in reduced_vars)
        other_subs = tuple((k, v) for (k, v) in subs if v.omega.name not in reduced_vars)
        new_reduced_vars = reduced_vars - {v.omega.name for (k, v) in subs if v.omega.name in reduced_vars}
        new_reduced_vars |= {k for (k, v) in subs if v.omega.name in reduced_vars}
        return (log_density + Subs(arg.arg, other_subs)).reduce(op, new_reduced_vars)
    return None


#####################################################################
# Handling invertible nonlinear unary transforms of RandomVariables
#####################################################################

@eager.register(Unary, TransformOp, RandomVariable)
def eager_unary_rv(op, arg):
    raise NotImplementedError("TODO implement invertible transforms of RandomVariables")


#####################################################################
# Handling linear/affine transformations of single RandomVariables
#####################################################################

@eager.register(Binary, AssociativeOp, NormalRV, (Number, Tensor))
def eager_binary_normalrv_const(op, rv, const):
    if op is ops.add:
        return NormalRV(rv.loc + const, rv.scale, omega=rv.omega)
    if op is ops.sub:
        return rv + -const
    if op is ops.mul:  # TODO handle negatives
        return NormalRV(rv.loc * const, rv.scale * ops.abs(const), omega=rv.omega)
    if op is ops.truediv:
        return rv * (1./const)
    return None  # TODO implement any missing binary operations (matmul?)


@eager.register(Binary, AssociativeOp, (Number, Tensor), NormalRV)
def eager_binary_const_normalrv(op, const, rv):
    if op is ops.add or op is ops.mul:
        return op(rv, const)
    return None  # TODO implement any missing ops (ops.sub? ops.getitem?)


@eager.register(Unary, AssociativeOp, NormalRV)
def eager_unary_neg(op, rv):
    if op is ops.sub:
        return NormalRV(-rv.loc, rv.scale, omega=rv.omega)
    return None  # TODO implement any missing unary operations


#######################################################
# Handling combinations of RandomVariables
#######################################################

@eager.register(Binary, AssociativeOp, NormalRV, NormalRV)
def eager_binary_normal_normal(op, lhs, rhs):
    if lhs.omega.name == rhs.omega.name:
        raise NotImplementedError("TODO alpha-convert")
    if op is ops.add:
        new_scale = ops.sqrt(lhs.scale ** 2 + rhs.scale ** 2)
        return NormalRV(lhs.loc + rhs.loc, new_scale, omega=lhs.omega)
    if op is ops.sub:
        return lhs + -rhs
    return None
