from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from six import add_metaclass

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
            # Integrate out delayed discrete variables.
            discrete_vars = reduced_vars.intersection(self.discrete.inputs)
            mixture_params = frozenset(self.gaussian.inputs).union(*(x.point.inputs for x in self.deltas))
            lazy_vars = discrete_vars & mixture_params  # Mixtures must remain lazy.
            discrete_vars -= mixture_params
            discrete = self.discrete.reduce(op, discrete_vars)

            # Integrate out delayed gaussian variables.
            gaussian_vars = reduced_vars.intersection(self.gaussian.inputs)
            gaussian = self.gaussian.reduce(ops.logaddexp, gaussian_vars)
            assert (reduced_vars - gaussian_vars).issubset(d.name for d in self.deltas)

            # Integrate out delayed degenerate variables, i.e. drop them.
            deltas = tuple(d for d in self.deltas if d.name not in reduced_vars)

            assert not lazy_vars
            return (Joint(deltas, discrete) + gaussian).reduce(ops.logaddexp, lazy_vars)

        if op is ops.add:
            raise NotImplementedError('TODO product-reduce along a plate dimension')

        return None  # defer to default implementation


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


@eager.register(Binary, AddOp, Joint, Gaussian)
def eager_add(op, joint, other):
    # Update with a delayed gaussian random variable.
    subs = tuple((d.name, d.point) for d in joint.deltas if d in other.inputs)
    if subs:
        other = other.eager_subs(subs)
    if joint.gaussian is not Number(0):
        other = joint.gaussian + other
    if not isinstance(other, Gaussian):
        return Joint(joint.deltas, joint.discrete) + other
    return Joint(joint.deltas, joint.discrete, other)


@eager.register(Binary, AddOp, Joint, Binary)
def eager_add(op, joint, other):
    if other.op is op:
        # Recursively decompose sums.
        joint += other.lhs
        joint += other.rhs
        return joint

    return None  # defer to default implementation


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
