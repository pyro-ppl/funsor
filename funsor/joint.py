from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

from six import add_metaclass
from six.moves import reduce

import funsor.interpreter as interpreter
import funsor.ops as ops
import funsor.terms
from funsor.delta import Delta
from funsor.domains import reals
from funsor.gaussian import Gaussian, sym_inverse
from funsor.integrate import Integrate, integrator
from funsor.montecarlo import monte_carlo
from funsor.ops import AddOp, NegOp, SubOp
from funsor.terms import (
    Align,
    Binary,
    Funsor,
    FunsorMeta,
    Independent,
    Number,
    Reduce,
    Subs,
    Unary,
    Variable,
    eager,
    to_funsor
)
from funsor.torch import Tensor, arange


class JointMeta(FunsorMeta):
    """
    Wrapper to fill in defaults and convert to funsor.
    """
    def __call__(cls, deltas=(), discrete=0, gaussian=0):
        discrete = to_funsor(discrete)
        gaussian = to_funsor(gaussian)
        return super(JointMeta, cls).__call__(deltas, discrete, gaussian)


@add_metaclass(JointMeta)
class Joint(Funsor):
    """
    Normal form for a joint log probability density funsor.

    The primary purpose of Joint is to handle substitution of
    :class:`~funsor.delta.Delta` funsors into other funsors.

    Joint is closed under Bayesian fusion, i.e. under ``ops.add`` operations.
    Joint is not closed under mixtures, i.e. ``ops.logaddexp`` operations,
    hence mixtures will be represented as lazy ``ops.logaddexp`` of Joints.

    :param tuple deltas: A possibly-empty tuple of degenerate distributions
        represented as :class:`~funsor.delta.Delta` funsors.
    :param Funsor discrete: A joint discrete log mass function represented as
        a :class:`~funsor.terms.Number` or `~funsor.terms.Tensor`.
    :param Funsor gaussian: An optional joint multivariate normal distribution
        a represented as :class:`~funsor.gaussian.Gaussian` or ``Number(0)`` if
        absent.
    """
    def __init__(self, deltas, discrete, gaussian):
        assert isinstance(deltas, tuple)
        assert isinstance(discrete, (Number, Tensor))
        assert discrete.output == reals()
        assert gaussian is Number(0) or isinstance(gaussian, Gaussian)
        inputs = OrderedDict()
        for x in deltas:
            assert isinstance(x, Delta)
            assert x.name not in inputs
            assert x.name not in discrete.inputs
            assert x.name not in gaussian.inputs
            inputs.update(x.inputs)
        inputs.update(discrete.inputs)
        inputs.update(gaussian.inputs)
        output = reals()
        super(Joint, self).__init__(inputs, output)
        self.deltas = deltas
        self.discrete = discrete
        self.gaussian = gaussian

    def eager_subs(self, subs):
        discrete = Subs(self.discrete, subs)
        gaussian = Subs(self.gaussian, subs)
        deltas = []
        for x in self.deltas:
            x = Subs(x, subs)
            if isinstance(x, Delta):
                deltas.append(x)
            elif isinstance(x, (Number, Tensor)):
                discrete += x
            else:
                raise ValueError('Cannot substitute {}'.format(x))
        deltas = tuple(deltas)
        return Joint(deltas, discrete) + gaussian

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            # Keep mixture parameters lazy.
            mixture_vars = frozenset(k for k, d in self.gaussian.inputs.items() if d.dtype != 'real')
            mixture_vars = mixture_vars.union(*(x.point.inputs for x in self.deltas))
            lazy_vars = reduced_vars & mixture_vars
            reduced_vars -= lazy_vars

            # Integrate out degenerate variables, i.e. drop selected delta.
            deltas = []
            remaining_vars = set(reduced_vars)
            for d in self.deltas:
                if d.name in reduced_vars:
                    remaining_vars.remove(d.name)
                else:
                    deltas.append(d)
            deltas = tuple(deltas)
            reduced_vars = frozenset(remaining_vars)

            # Integrate out delayed discrete variables.
            discrete_vars = reduced_vars.intersection(self.discrete.inputs)
            discrete = self.discrete.reduce(op, discrete_vars)
            reduced_vars -= discrete_vars

            # Integrate out delayed gaussian variables.
            gaussian_vars = reduced_vars.intersection(self.gaussian.inputs)
            gaussian = self.gaussian.reduce(ops.logaddexp, gaussian_vars)
            reduced_vars -= gaussian_vars

            # Scale to account for remaining reduced_vars that were inputs to dropped deltas.
            eager_result = Joint(deltas, discrete)
            if gaussian is not Number(0):
                eager_result += gaussian
            reduced_vars |= lazy_vars.difference(eager_result.inputs)
            lazy_vars = lazy_vars.intersection(eager_result.inputs)
            if reduced_vars:
                eager_result += ops.log(reduce(ops.mul, [self.inputs[v].dtype for v in reduced_vars]))

            # Return a value only if progress has been made.
            if eager_result is self:
                return None  # defer to default implementation
            else:
                return eager_result.reduce(ops.logaddexp, lazy_vars)

        if op is ops.add:
            terms = list(self.deltas) + [self.discrete, self.gaussian]
            for i, term in enumerate(terms):
                terms[i] = term.reduce(ops.add, reduced_vars.intersection(term.inputs))
            return reduce(ops.add, terms)

        return None  # defer to default implementation

    def moment_matching_reduce(self, op, reduced_vars):
        if not reduced_vars:
            return self
        if op is ops.logaddexp:
            if not all(reduced_vars.isdisjoint(d.inputs) for d in self.deltas):
                raise NotImplementedError('TODO handle moment_matching with Deltas')
            # FIXME fix partitioning of variables.
            approx_vars = frozenset(k for k in reduced_vars if self.inputs[k].dtype != 'real')
            exact_vars = frozenset(k for k in reduced_vars if self.inputs[k].dtype == 'real')
            if exact_vars:
                return self.reduce(op, exact_vars).reduce(op, approx_vars)

            # Moment-matching approximation.
            assert approx_vars and not exact_vars
            discrete = self.discrete
            gaussian = self.gaussian
            if not approx_vars.issubset(discrete.inputs):
                raise NotImplementedError('TODO')
            if not approx_vars.issubset(gaussian.inputs):
                raise NotImplementedError('TODO')
            int_inputs = OrderedDict((k, d) for k, d in gaussian.inputs.items()
                                     if d.dtype != 'real')
            if discrete.inputs != int_inputs:
                raise NotImplementedError('TODO')
            old_loc = Tensor(gaussian.loc, int_inputs)
            new_discrete = discrete.reduce(ops.logaddexp, approx_vars)
            probs = (discrete - new_discrete).exp()
            new_loc = (probs * old_loc).reduce(ops.add, approx_vars)
            old_cov = Tensor(sym_inverse(gaussian.precision), int_inputs)
            diffs = (old_loc - new_loc).data
            outers = Tensor(diffs.unsqueeze(-1) * diffs.unsqueeze(-2), int_inputs)
            new_cov = ((probs * old_cov).reduce(ops.add, approx_vars) +
                       (probs * outers).reduce(ops.add, approx_vars)).data
            new_precision = sym_inverse(new_cov)
            new_inputs = OrderedDict((k, d) for k, d in gaussian.inputs.items()
                                     if k not in approx_vars)
            new_gaussian = Gaussian(new_loc.data, new_precision, new_inputs)
            return Joint(self.deltas, new_discrete, new_gaussian)

        return None  # defer to default implementation

    def unscaled_sample(self, sampled_vars, sample_inputs):
        discrete_vars = sampled_vars.intersection(self.discrete.inputs)
        gaussian_vars = frozenset(k for k, v in self.gaussian.inputs.items()
                                  if k in sampled_vars if v.dtype == 'real')
        result = self
        if discrete_vars:
            discrete = result.discrete.unscaled_sample(discrete_vars, sample_inputs)
            result = Joint(result.deltas, gaussian=result.gaussian) + discrete
        if gaussian_vars:
            # Draw an expanded sample.
            gaussian_sample_inputs = sample_inputs.copy()
            for k, d in self.inputs.items():
                if k not in result.gaussian.inputs and d.dtype != 'real':
                    gaussian_sample_inputs[k] = d
            gaussian = result.gaussian.unscaled_sample(gaussian_vars, gaussian_sample_inputs)
            result = Joint(result.deltas, result.discrete) + gaussian
        return result


@eager.register(Joint, tuple, Funsor, Funsor)
def eager_joint(deltas, discrete, gaussian):
    # Demote a Joint to a simpler elementart funsor.
    if not deltas:
        if gaussian is Number(0):
            return discrete
        elif discrete is Number(0):
            return gaussian
    elif len(deltas) == 1:
        if discrete is Number(0) and gaussian is Number(0):
            return deltas[0]

    return None  # defer to default implementation


@eager.register(Independent, Joint, str, str)
def eager_independent(joint, reals_var, bint_var):
    for i, delta in enumerate(joint.deltas):
        if delta.name == reals_var:
            delta = Independent(delta, reals_var, bint_var)
            deltas = joint.deltas[:i] + (delta,) + joint.deltas[1+i:]
            discrete = joint.discrete
            if bint_var in discrete.inputs:
                discrete = discrete.reduce(ops.add, bint_var)
            gaussian = joint.gaussian
            if bint_var in gaussian.inputs:
                gaussian = gaussian.reduce(ops.add, bint_var)
            return Joint(deltas, discrete, gaussian)

    return None  # defer to default implementation


################################################################################
# Patterns to update a Joint with other funsors
################################################################################

@eager.register(Binary, AddOp, Joint, Joint)
def eager_add(op, joint, other):
    # Fuse two joint distributions.
    for d in other.deltas:
        joint += d
    joint += other.discrete
    joint += other.gaussian
    return joint


@eager.register(Binary, AddOp, Joint, Delta)
def eager_add(op, joint, delta):
    # Update with a degenerate distribution, typically a monte carlo sample.
    if delta.name in joint.inputs:
        joint = Subs(joint, ((delta.name, delta.point),))
        if not isinstance(joint, Joint):
            return joint + delta
    for d in joint.deltas:
        if d.name in delta.inputs:
            delta = Subs(delta, ((d.name, d.point),))
    deltas = joint.deltas + (delta,)
    return Joint(deltas, joint.discrete, joint.gaussian)


@eager.register(Binary, AddOp, Joint, (Number, Tensor))
def eager_add(op, joint, other):
    # Update with a delayed discrete random variable.
    subs = tuple((d.name, d.point) for d in joint.deltas if d in other.inputs)
    if subs:
        return joint + Subs(other, subs)
    return Joint(joint.deltas, joint.discrete + other, joint.gaussian)


@eager.register(Binary, AddOp, Joint, Gaussian)
def eager_add(op, joint, other):
    # Update with a delayed gaussian random variable.
    subs = tuple((d.name, d.point) for d in joint.deltas if d.name in other.inputs)
    if subs:
        other = Subs(other, subs)
    if joint.gaussian is not Number(0):
        other = joint.gaussian + other
    if not isinstance(other, Gaussian):
        return Joint(joint.deltas, joint.discrete) + other
    return Joint(joint.deltas, joint.discrete, other)


eager.register(Binary, AddOp, Reduce, Joint)(
    funsor.terms.eager_distribute_reduce_other)


@eager.register(Binary, AddOp, (Funsor, Align, Delta), Joint)
def eager_add(op, other, joint):
    return joint + other


################################################################################
# Patterns to create a Joint from elementary funsors
################################################################################

@eager.register(Binary, AddOp, Delta, Delta)
def eager_add(op, lhs, rhs):
    if lhs.name == rhs.name:
        raise NotImplementedError
    if rhs.name in lhs.inputs:
        assert lhs.name not in rhs.inputs
        lhs = lhs(**{rhs.name: rhs.point})
    elif lhs.name in rhs.inputs:
        rhs = rhs(**{lhs.name: lhs.point})
    return Joint(deltas=(lhs, rhs))


@eager.register(Binary, AddOp, Delta, (Number, Tensor, Gaussian))
def eager_add(op, delta, other):
    if delta.name in other.inputs:
        other = Subs(other, ((delta.name, delta.point),))
        assert isinstance(other, (Number, Tensor, Gaussian))
    if isinstance(other, (Number, Tensor)):
        return Joint((delta,), discrete=other)
    else:
        return Joint((delta,), gaussian=other)


@eager.register(Binary, AddOp, (Number, Tensor, Gaussian), Delta)
def eager_add(op, other, delta):
    return delta + other


@eager.register(Binary, AddOp, Gaussian, (Number, Tensor))
def eager_add(op, gaussian, discrete):
    return Joint(discrete=discrete, gaussian=gaussian)


@eager.register(Binary, AddOp, (Number, Tensor), Gaussian)
def eager_add(op, discrete, gaussian):
    return Joint(discrete=discrete, gaussian=gaussian)


################################################################################
# Patterns to compute Radon-Nikodym derivatives
################################################################################

@eager.register(Binary, SubOp, Joint, (Funsor, Align, Gaussian, Joint))
def eager_sub(op, joint, other):
    return joint + -other


@eager.register(Binary, SubOp, (Funsor, Align), Joint)
def eager_sub(op, other, joint):
    return -joint + other


@eager.register(Binary, SubOp, Delta, (Number, Tensor, Gaussian, Joint))
@eager.register(Binary, SubOp, (Number, Tensor), Gaussian)
@eager.register(Binary, SubOp, Gaussian, (Number, Tensor, Joint))
def eager_sub(op, lhs, rhs):
    return lhs + -rhs


@eager.register(Unary, NegOp, Joint)
def eager_neg(op, joint):
    if joint.deltas:
        raise ValueError("Cannot negate deltas")
    discrete = -joint.discrete
    gaussian = -joint.gaussian
    return Joint(discrete=discrete, gaussian=gaussian)


################################################################################
# Patterns for integration
################################################################################

def _simplify_integrate(fn, joint, integrand, reduced_vars):
    if any(d.name in reduced_vars for d in joint.deltas):
        subs = tuple((d.name, d.point) for d in joint.deltas if d.name in reduced_vars)
        deltas = tuple(d for d in joint.deltas if d.name not in reduced_vars)
        log_measure = Joint(deltas, joint.discrete, joint.gaussian)
        integrand = Subs(integrand, subs)
        reduced_vars = reduced_vars - frozenset(name for name, point in subs)
        return Integrate(log_measure, integrand, reduced_vars)

    return fn(log_measure, integrand, reduced_vars)


def _joint_integrator(fn):
    """
    Decorator for Integrate(Joint(...), ...) patterns.
    """
    fn = interpreter.debug_logged(fn)
    return integrator(functools.partial(_simplify_integrate, fn))


@eager.register(Integrate, Joint, Funsor, frozenset)
@_joint_integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    return None  # defer to default implementation


@eager.register(Integrate, Joint, Delta, frozenset)
@_joint_integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    raise NotImplementedError('TODO')


@eager.register(Integrate, Joint, Tensor, frozenset)
@_joint_integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    raise NotImplementedError('TODO')


@eager.register(Integrate, Joint, Gaussian, frozenset)
@_joint_integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    raise NotImplementedError('TODO')


@eager.register(Integrate, Joint, Joint, frozenset)
@_joint_integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    raise NotImplementedError('TODO')


@eager.register(Integrate, Joint, Variable, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    name = integrand.name
    assert reduced_vars == frozenset([name])
    if any(d.name == name for d in log_measure.deltas):
        deltas = tuple(d for d in log_measure.deltas if d.name != name)
        log_norm = Joint(deltas, log_measure.discrete, log_measure.gaussian)
        for d in log_measure.deltas:
            if d.name == name:
                mean = d.point
                break
        return mean * log_norm.exp()
    elif name in log_measure.discrete.inputs:
        integrand = arange(name, integrand.inputs[name].dtype)
        return Integrate(log_measure, integrand, reduced_vars)
    else:
        assert name in log_measure.gaussian.inputs
        gaussian = Integrate(log_measure.gaussian, integrand, reduced_vars)
        return Joint(log_measure.deltas, log_measure.discrete).exp() * gaussian


@monte_carlo.register(Integrate, Joint, Funsor, frozenset)
@integrator
def monte_carlo_integrate(log_measure, integrand, reduced_vars):
    sampled_log_measure = log_measure.sample(reduced_vars, monte_carlo.sample_inputs)
    if sampled_log_measure is not log_measure:
        reduced_vars = reduced_vars | frozenset(monte_carlo.sample_inputs)
        return Integrate(sampled_log_measure, integrand, reduced_vars)

    return None  # defer to default implementation


__all__ = [
    'Joint',
]
