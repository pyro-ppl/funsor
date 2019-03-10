from __future__ import absolute_import, division, print_function

from six import add_metaclass

import funsor.ops as ops
from funsor.domains import reals
from funsor.gaussian import Gaussian
from funsor.ops import AssociativeOp
from funsor.terms import Binary, Funsor, FunsorMeta, Number, eager, to_funsor
from funsor.torch import Tensor
from funsor.delta import Delta


class JointNormalFormMeta(FunsorMeta):
    """
    Wrapper to fill in default values.
    """
    def __call__(cls, gaussian=0, deltas=()):
        gaussian = to_funsor(gaussian)
        return super(JointNormalForm, cls).__call__(gaussian, deltas)


@add_metaclass(JointNormalFormMeta)
class JointNormalForm(Funsor):
    """
    Normal form for a joint log probability density.
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
                deltas.append(x)
            elif isinstance(x, (Number, Tensor)):
                gaussian += x
            else:
                raise ValueError('Cannot substitute {}'.format(x))
        deltas = tuple(deltas)
        return JointNormalForm(gaussian, deltas)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if reduced_vars == frozenset(self.inputs):
                return self.gaussian.eager_reduce(reduced_vars)
            for var in reduced_vars:
                if var in self.gaussian.inputs and self.inputs[var].dtype != 'real':
                    raise ValueError('Mixture distributions are not supported')
            raise NotImplementedError('TODO eliminate a random variable')

        if op is ops.add:
            raise NotImplementedError('TODO eliminate a plate')

        return None  # defer to default implementation


@eager.register(Binary, AssociativeOp, JointNormalForm, (Number, Tensor, Gaussian))
def eager_binary_joint_gaussian(op, joint, gaussian):
    if op is ops.add:
        # Accumulate a lazy Categorical or Normal random variable.
        for d in joint.deltas:
            if d.name in gaussian.inputs:
                gaussian += d
        gaussian = joint.gaussian + gaussian
        return JointNormalForm(gaussian, joint.deltas)

    return None  # defer to default implementation


@eager.register(Binary, AssociativeOp, JointNormalForm, Delta)
def eager_binary_joint_delta(op, joint, gaussian):
    if op is ops.add:
        raise NotImplementedError('TODO accumulate a monte carlo sample')

    return None  # defer to default implementation


__all__ = [
    'JointNormalForm',
]
