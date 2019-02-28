from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch

from funsor.domains import bint, reals
from funsor.gaussian import Gaussian


def test_smoke():
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
    Gaussian(log_density, loc, scale_tril, inputs)
