from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.torch import Tensor
from funsor.terms import Number


@pytest.mark.parametrize('expr,expected_type', [
    ('g1 + 1', Gaussian),
    ('g1 - 1', Gaussian),
    ('1 + g1', Gaussian),
    ('g1 + shift', Gaussian),
    ('g1 - shift', Gaussian),
    ('shift + g1', Gaussian),
    ('g1 + g1', Gaussian),
    ('g1(i=i0)', Gaussian),
    ('g2(i=i0)', Gaussian),
    ('g1(i=i0) + g2(i=i0)', Gaussian),
    ('g1(i=i0) + g2', Gaussian),
    ('g1(x=x0)', Tensor),
    ('g2(y=y0)', Tensor),
    ('(g1 + g2)(i=i0)', Gaussian),
    ('(g1 + g2)(x=x0, y=y0)', Tensor),
    ('g1.logsumexp("x")', Tensor),
    # ('(g1 + g2).logsumexp("x")', Gaussian),
    # ('(g1 + g2).logsumexp("y")', Gaussian),
    ('(g1 + g2).logsumexp(frozenset(["x", "y"]))', Tensor),
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

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    y0 = Tensor(torch.tensor([[0.2, 0.3],
                              [0.8, 0.9]]),
                inputs=OrderedDict([('i', bint(2))]))
    assert isinstance(y0, Tensor)

    result = eval(expr)
    assert isinstance(result, expected_type)
