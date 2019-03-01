from __future__ import absolute_import, division, print_function

from six import integer_types

from funsor.terms import Funsor, Number, Variable
from funsor.torch import Tensor, arange


class Affine(Funsor):
    def __init__(self, constant, linear):
        assert isinstance(constant, Funsor)
        assert constant.dtype == 'real'
        assert all(d.dtype != 'real' for d in constant.inputs.values())
        assert isinstance(linear, tuple)
        output = constant.output
        inputs = constant.inputs.copy()
        for coeff, var in linear:
            assert isinstance(coeff, Funsor)
            assert all(d.dtype != 'real' for d in coeff.inputs.values())
            assert isinstance(var, Variable)
            assert coeff.dtype == 'real'
            assert coeff.shape == output.shape + var.output.shape  # a linear function
            assert var.dtype == 'real'
            inputs.update(coeff.inputs)
            inputs.update(var.inputs)
        return super(Affine, self).__init__(inputs, output)

    def eager_subs(self, subs):
        raise NotImplementedError


def to_affine(x):
    """
    Attempt to convert a Funsor to an :class:`Affine` funsor of form

        a + (b1 * v1 + b2 * v2 + ... + bn * vn)

    where ``a``,``b1``, ..., ``bn`` are :class:`~funsor.terms.Number`s or
    :class:`~funsor.torch.Tensor`~, and ``v1``, ..., ``vn`` are real
    :class:`~funsor.terms.Variable`s.

    On failure, this raises a ``ValueError``.

    :param Funsor x:
    :rtype: Affine
    :param
    :raises: ValueError
    """
    assert isinstance(x, Funsor)
    if isinstance(x, (Number, Tensor)):
        return x
    subs = []
    for name, domain in x.inputs.items():
        if isinstance(domain.dtype, integer_types):
            subs.append((name, arange(name, domain.dtype)))
    subs = tuple(subs)
    x = x.eager_subs(subs)

    raise NotImplementedError('TODO, match patterns')

    return x


__all__ = [
    'Affine',
    'to_affine',
]
