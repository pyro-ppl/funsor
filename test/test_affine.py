from __future__ import absolute_import, division, print_function

import pytest
import torch

from funsor.affine import Affine, to_affine
from funsor.domains import reals
from funsor.terms import Number, Variable
from funsor.torch import Tensor

TENSORS = {
    'x_34': torch.randn(3, 4),
}


@pytest.mark.xfail(reason='TODO')
@pytest.mark.parametrize('x,constant,linear', [
    (Number(0), Number(0), ()),
    (Tensor(TENSORS['x_34']), Tensor(TENSORS['x_34']), ()),
    (Variable('x', reals()), Number(0), ((Number(1), Variable('x', reals())))),
])
def test_to_affine(x, constant, linear):
    actual = to_affine(x)
    expected = Affine(constant, linear)
    assert actual is expected
