from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.torch import Tensor


@pytest.mark.parametrize('expr,expected_type', [
    ('g.logsumexp("x")', Tensor),
    ('g + 1', Gaussian),
    ('1 + g', Gaussian),
    ('g + shift', Gaussian),
    ('shift + g', Gaussian),
    ('g + g', Gaussian),
])
def test_smoke(expr, expected_type):
    log_density = torch.tensor([0.0, 1.0])
    loc = torch.tensor([[0.0, 0.1, 0.2],
                        [2.0, 3.0, 4.0]])
    scale_tril = torch.tensor([[[1.0, 0.0, 0.0],
                                [0.1, 1.0, 0.0],
                                [0.2, 0.3, 1.0]],
                               [[1.0, 0.0, 0.0],
                                [0.1, 1.0, 0.0],
                                [0.2, 0.3, 1.0]]])
    inputs = OrderedDict([('i', bint(2)), ('x', reals(3))])
    g = Gaussian(log_density, loc, scale_tril, inputs)
    assert isinstance(g, Gaussian)
    shift = Tensor(torch.tensor([-1., 1.]), OrderedDict([('i', bint(2))]))
    assert isinstance(shift, Tensor)

    result = eval(expr)
    assert isinstance(result, expected_type)
