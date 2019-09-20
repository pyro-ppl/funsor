from collections import OrderedDict

import opt_einsum
import torch

from funsor.interpreter import gensym
from funsor.terms import Lambda, Variable, bint
from funsor.torch import Tensor


def extract_affine(fn):
    """
    Returns a pair ``(const, coeffs)`` where const is a funsor with no real
    inputs and ``coeffs`` is an OrderedDict mapping input name to a
    ``(coefficient, eqn)`` pair in einsum form. For funsors that are affine,
    this satisfies::

        x = Contraction(...)
        assert x.is_affine
        const, coeffs = x.extract_affine()
        y = sum(Einsum(eqn, (coeff, Variable(var, coeff.output)))
                for var, (coeff, eqn) in coeffs.items())
        assert_close(y, x)

    :param Funsor fn:
    """
    # Avoid adding these dependencies to funsor core.
    real_inputs = OrderedDict((k, v) for k, v in fn.inputs.items() if v.dtype == 'real')
    coeffs = OrderedDict()
    zeros = {k: Tensor(torch.zeros(v.shape)) for k, v in real_inputs.items()}
    const = fn(**zeros)
    name = gensym('probe')
    for k, v in real_inputs.items():
        dim = v.num_elements
        var = Variable(name, bint(dim))
        subs = zeros.copy()
        subs[k] = Tensor(torch.eye(dim).reshape((dim,) + v.shape))[var]
        coeff = Lambda(var, fn(**subs) - const).reshape(v.shape + const.shape)
        inputs1 = ''.join(map(opt_einsum.get_symbol, range(len(coeff.shape))))
        inputs2 = inputs1[:len(v.shape)]
        output = inputs1[len(v.shape):]
        eqn = f"{inputs1},{inputs2}->{output}"
        coeffs[k] = coeff, eqn
    return const, coeffs
