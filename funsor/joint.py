from __future__ import absolute_import, division, print_function

from six import add_metaclass

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import reals
from funsor.gaussian import Gaussian
from funsor.ops import AddOp
from funsor.terms import Binary, Funsor, FunsorMeta, Number, eager, to_funsor
from funsor.torch import Tensor


class JointMeta(FunsorMeta):
    """
    Wrapper to fill in defaults and move log_density terms into gaussian.
    """
    def __call__(cls, gaussian=0, deltas=()):
        gaussian = to_funsor(gaussian)

        # Move all log_density terms into gaussian part.
        deltas = list(deltas)
        for i, d in enumerate(deltas):
            if d.log_density is not Number(0):
                gaussian += d.log_density
                deltas[i] = Delta(d.name, d.point)
        deltas = tuple(deltas)

        return super(Joint, cls).__call__(gaussian, deltas)


@add_metaclass(JointMeta)
class Joint(Funsor):
    """
    Normal form for a joint log probability density funsor.
    """
    def __init__(self, gaussian, deltas):
        assert isinstance(gaussian, (Number, Tensor, Gaussian))
        assert isinstance(deltas, tuple)
        inputs = gaussian.inputs.copy()
        for x in deltas:
            assert isinstance(x, Delta)
            assert x.name not in inputs
            inputs.update(x.inputs)
        output = reals()
        super(Funsor, self).__init__(self, inputs, output)
        self.gaussian = gaussian
        self.deltas = deltas

    def eager_subs(self, subs):
        gaussian = self.gaussian.eager_subs(subs)
        assert isinstance(gaussian, (Number, Tensor, Gaussian))
        deltas = []
        for x in self.deltas:
            x = x.eager_subs(subs)
            if isinstance(x, Delta):
                assert x.log_density is Number(0)
                deltas.append(x)
            elif isinstance(x, (Number, Tensor)):
                gaussian += x
            else:
                raise ValueError('Cannot substitute {}'.format(x))
        deltas = tuple(deltas)
        return Joint(gaussian, deltas)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            # Integrate out a delayed variable.
            gaussian_vars = reduced_vars.intersection(self.gaussian.inputs)
            gaussian = self.gaussian.reduce(ops.logaddexp, gaussian_vars)
            assert (reduced_vars - gaussian_vars).issubset(d.name for d in self.deltas)
            deltas = tuple(d for d in self.deltas if d.name not in reduced_vars)
            return Joint(gaussian, deltas)

        if op is ops.sample:
            # Sample a delayed variable.
            # FIXME how do we retrieve the variable value?
            assert reduced_vars.issubset(self.gaussian.inputs)
            gaussian = self.gaussian.reduce(ops.sample, reduced_vars)
            return Joint(0, self.deltas) + gaussian

        if op is ops.add:
            # Eliminate a plate.
            raise NotImplementedError('TODO')

        return None  # defer to default implementation


@eager.register(Binary, AddOp, Joint, (Number, Tensor, Gaussian))
def eager_add(op, joint, gaussian):
    # Add a log_prob term for a delayed factor.
    gaussian = gaussian(**{d.name: d.point for d in joint.deltas})
    gaussian = joint.gaussian + gaussian
    return Joint(gaussian, joint.deltas)


@eager.register(Binary, AddOp, Joint, Delta)
def eager_add(op, joint, delta):
    # Add an eagerly sampled value.
    joint = joint(**{delta.name: delta.point})
    # FIXME this can miss some substitutions of self into delta
    return Joint(joint.gaussian, joint.deltas + (delta,))


@eager.register(Binary, AddOp, Joint, Joint)
def eager_add(op, joint, other):
    # Fuse two joint distributions.
    for d in other.deltas:
        joint += d
    joint += other.gaussian
    return joint


@eager.register(Binary, AddOp, Joint, Binary)
def eager_add(op, joint, other):
    if other.op is op:
        # Recursively decompose sums.
        joint += other.lhs
        joint += other.rhs
        return joint

    return None  # defer to default implementation


@eager.register(Binary, AddOp, Funsor, Joint)
def eager_add(op, other, joint):
    return joint + other


@eager.register(Binary, AddOp, (Number, Tensor, Gaussian), Delta)
def eager_add(op, gaussian, delta):
    gaussian = gaussian(**{delta.name: delta.point})
    return Joint(gaussian, (delta,))


@eager.register(Binary, AddOp, Delta, (Number, Tensor, Gaussian))
def eager_add(op, delta, gaussian):
    gaussian = gaussian(**{delta.name: delta.point})
    return Joint(gaussian, (delta,))


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


__all__ = [
    'Joint',
]
