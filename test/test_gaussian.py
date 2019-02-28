from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.torch import Tensor


@pytest.mark.parametrize('expr,expected_type', [
    ('g1.logsumexp("x")', Tensor),
    ('g1 + 1', Gaussian),
    ('1 + g1', Gaussian),
    ('g1 + shift', Gaussian),
    ('shift + g1', Gaussian),
    ('g1 + g1', Gaussian),
    ('g1 + g2', Gaussian),
])
def test_smoke(expr, expected_type):
    g1 = Gaussian(
        log_density=torch.tensor([0.0, 1.0]),
        loc=torch.tensor([[0.0, 0.1, 0.2],
                          [2.0, 3.0, 4.0]]),
        scale_tril=torch.tensor([[[1.0, 0.0, 0.0],
                                  [0.1, 1.0, 0.0],
                                  [0.2, 0.3, 1.0]],
                                 [[1.0, 0.0, 0.0],
                                  [0.1, 1.0, 0.0],
                                  [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('x', reals(3))]))
    assert isinstance(g1, Gaussian)

    g2 = Gaussian(
        log_density=torch.tensor([0.0, 1.0]),
        loc=torch.tensor([[0.0, 0.1],
                          [2.0, 3.0]]),
        scale_tril=torch.tensor([[[1.0, 0.0],
                                  [0.2, 1.0]],
                                 [[1.0, 0.0],
                                  [0.2, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('y', reals(2))]))
    assert isinstance(g2, Gaussian)

    shift = Tensor(torch.tensor([-1., 1.]), OrderedDict([('i', bint(2))]))
    assert isinstance(shift, Tensor)

    result = eval(expr)
    assert isinstance(result, expected_type)
