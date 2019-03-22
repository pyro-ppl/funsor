from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from six import add_metaclass
from six.moves import reduce

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import reals
from funsor.gaussian import Gaussian
from funsor.ops import AddOp, Op
from funsor.terms import Align, Binary, Funsor, FunsorMeta, Number, eager, to_funsor
from funsor.torch import Tensor


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
        gaussian = self.gaussian.eager_subs(subs)
        assert isinstance(gaussian, (Number, Tensor, Gaussian))
        discrete = self.discrete.eager_subs(subs)
        gaussian = self.gaussian.eager_subs(subs)
        deltas = []
        for x in self.deltas:
            x = x.eager_subs(subs)
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
            eager_result = Joint(deltas, discrete) + gaussian
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
            raise NotImplementedError('TODO product-reduce along a plate dimension')

        return None  # defer to default implementation

    def sample(self, sampled_vars, sample_inputs=None):
        if sample_inputs is None:
            sample_inputs = OrderedDict()
        assert frozenset(sample_inputs).isdisjoint(self.inputs)
        discrete_vars = sampled_vars.intersection(self.discrete.inputs)
        gaussian_vars = frozenset(k for k in sampled_vars
                                  if k in self.gaussian.inputs
                                  if self.gaussian.inputs[k].dtype == 'real')
        result = self
        if discrete_vars:
            discrete = result.discrete.sample(discrete_vars, sample_inputs)
            result = Joint(result.deltas, gaussian=result.gaussian) + discrete
        if gaussian_vars:
            sample_inputs = OrderedDict((k, v) for k, v in sample_inputs.items()
                                        if k not in result.gaussian.inputs)
            gaussian = result.gaussian.sample(gaussian_vars, sample_inputs)
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
        joint = joint.eager_subs(((delta.name, delta.point),))
        if not isinstance(joint, Joint):
            return joint + delta
    for d in joint.deltas:
        if d.name in delta.inputs:
            delta = delta.eager_subs(((d.name, d.point),))
    deltas = joint.deltas + (delta,)
    return Joint(deltas, joint.discrete, joint.gaussian)


@eager.register(Binary, AddOp, Joint, (Number, Tensor))
def eager_add(op, joint, other):
    # Update with a delayed discrete random variable.
    subs = tuple((d.name, d.point) for d in joint.deltas if d in other.inputs)
    if subs:
        return joint + other.eager_subs(subs)
    return Joint(joint.deltas, joint.discrete + other, joint.gaussian)


@eager.register(Binary, Op, Joint, (Number, Tensor))
def eager_add(op, joint, other):
    if op is ops.sub:
        return joint + -other

    return None  # defer to default implementation


@eager.register(Binary, AddOp, Joint, Gaussian)
def eager_add(op, joint, other):
    # Update with a delayed gaussian random variable.
    subs = tuple((d.name, d.point) for d in joint.deltas if d.name in other.inputs)
    if subs:
        other = other.eager_subs(subs)
    if joint.gaussian is not Number(0):
        other = joint.gaussian + other
    if not isinstance(other, Gaussian):
        return Joint(joint.deltas, joint.discrete) + other
    return Joint(joint.deltas, joint.discrete, other)


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
        other = other.eager_subs(((delta.name, delta.point),))
        assert isinstance(other, (Number, Tensor, Gaussian))
    if isinstance(other, (Number, Tensor)):
        return Joint((delta,), discrete=other)
    else:
        return Joint((delta,), gaussian=other)


@eager.register(Binary, Op, Delta, (Number, Tensor))
def eager_binary(op, delta, other):
    if op is ops.sub:
        return delta + -other


@eager.register(Binary, AddOp, (Number, Tensor, Gaussian), Delta)
def eager_add(op, other, delta):
    return delta + other


@eager.register(Binary, AddOp, Gaussian, (Number, Tensor))
def eager_add(op, gaussian, discrete):
    return Joint(discrete=discrete, gaussian=gaussian)


@eager.register(Binary, AddOp, (Number, Tensor), Gaussian)
def eager_add(op, discrete, gaussian):
    return Joint(discrete=discrete, gaussian=gaussian)


@eager.register(Binary, Op, Gaussian, (Number, Tensor))
def eager_binary(op, gaussian, discrete):
    if op is ops.sub:
        return Joint(discrete=-discrete, gaussian=gaussian)


__all__ = [
    'Joint',
]
