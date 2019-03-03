from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.terms import Number
from funsor.testing import assert_close
from funsor.torch import Tensor


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
    ('(g2 + g1)(x=x0, y=y0)', Tensor),
    ('g1.logsumexp("x")', Tensor),
    ('(g1 + g2).logsumexp("x")', Gaussian),
    ('(g1 + g2).logsumexp("y")', Gaussian),
    ('(g1 + g2).logsumexp(frozenset(["x", "y"]))', Tensor),
])
def test_smoke(expr, expected_type):
    g1 = Gaussian(
        log_density=torch.tensor([0.0, 1.0]),
        loc=torch.tensor([[0.0, 0.1, 0.2],
                          [2.0, 3.0, 4.0]]),
        precision=torch.tensor([[[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]],
                                [[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('x', reals(3))]))
    assert isinstance(g1, Gaussian)

    g2 = Gaussian(
        log_density=torch.tensor([0.0, 1.0]),
        loc=torch.tensor([[0.0, 0.1],
                          [2.0, 3.0]]),
        precision=torch.tensor([[[1.0, 0.2],
                                 [0.2, 1.0]],
                                [[1.0, 0.2],
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


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
    {'i': bint(2), 'j': bint(3)},
], ids=lambda p: '-'.join(p.keys()))
@pytest.mark.parametrize('real_inputs', [
    {'x': reals()},
    {'x': reals(4)},
    {'x': reals(2, 3)},
    {'x': reals(), 'y': reals()},
    {'x': reals(2), 'y': reals(3)},
    {'x': reals(4), 'y': reals(2, 4), 'z': reals()},
], ids=lambda p: '-'.join(k + str(v.shape) for k, v in p.items()))
def test_binary_gaussian_number(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    batch_shape = tuple(domain.dtype for domain in int_inputs.values())
    event_shape = (sum(domain.num_elements for domain in real_inputs.values()),)
    log_density = torch.randn(batch_shape)
    loc = torch.randn(batch_shape + event_shape)
    prec_sqrt = torch.randn(batch_shape + event_shape + event_shape)
    precision = torch.matmul(prec_sqrt, prec_sqrt.transpose(-1, -2))
    g = Gaussian(log_density, loc, precision, inputs)
    n = Number(1.234)
    values = {name: Tensor(torch.randn(domain.shape))
              for name, domain in real_inputs.items()}

    assert_close((g + n)(**values), g(**values) + n)
    assert_close((n + g)(**values), n + g(**values))
    assert_close((g - n)(**values), g(**values) - n, atol=1e-4)
