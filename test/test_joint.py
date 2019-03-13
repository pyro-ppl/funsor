from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.joint import Joint
from funsor.terms import Number
from funsor.testing import assert_close, random_gaussian, random_tensor, xfail_if_not_implemented
from funsor.torch import Tensor


def id_from_inputs(inputs):
    if not inputs:
        return '()'
    return ','.join(k + ''.join(map(str, d.shape)) for k, d in inputs.items())


SMOKE_TESTS = [
    ('dx + dy', Joint),
    ('dx + g', Joint),
    ('dy + g', Joint),
    ('g + dx', Joint),
    ('g + dy', Joint),
    ('dx + t', Joint),
    ('dy + t', Joint),
    ('dx - t', Joint),
    ('dy - t', Joint),
    ('t + dx', Joint),
    ('t + dy', Joint),
    ('g + 1', Joint),
    ('g - 1', Joint),
    ('1 + g', Joint),
    ('g + t', Joint),
    ('g - t', Joint),
    ('t + g', Joint),
    ('g + g', Joint),
    ('(dx + dy)(i=i0)', Joint),
    ('(dx + g)(i=i0)', Joint),
    ('(dy + g)(i=i0)', Joint),
    ('(g + dx)(i=i0)', Joint),
    ('(g + dy)(i=i0)', Joint),
    ('(dx + t)(i=i0)', Joint),
    ('(dy + t)(i=i0)', Joint),
    ('(dx - t)(i=i0)', Joint),
    ('(dy - t)(i=i0)', Joint),
    ('(t + dx)(i=i0)', Joint),
    ('(t + dy)(i=i0)', Joint),
    ('(g + 1)(i=i0)', Joint),
    ('(g - 1)(i=i0)', Joint),
    ('(1 + g)(i=i0)', Joint),
    ('(g + t)(i=i0)', Joint),
    ('(g - t)(i=i0)', Joint),
    ('(t + g)(i=i0)', Joint),
    ('(g + g)(i=i0)', Joint),
    ('(dx + dy)(x=x0)', Joint),
    ('(dx + g)(x=x0)', Tensor),
    ('(dy + g)(x=x0)', Joint),
    ('(g + dx)(x=x0)', Tensor),
    ('(g + dy)(x=x0)', Joint),
    ('(dx + t)(x=x0)', Tensor),
    ('(dy + t)(x=x0)', Joint),
    ('(dx - t)(x=x0)', Tensor),
    ('(dy - t)(x=x0)', Joint),
    ('(t + dx)(x=x0)', Tensor),
    ('(t + dy)(x=x0)', Joint),
    ('(g + 1)(x=x0)', Tensor),
    ('(g - 1)(x=x0)', Tensor),
    ('(1 + g)(x=x0)', Tensor),
    ('(g + t)(x=x0)', Tensor),
    ('(g - t)(x=x0)', Tensor),
    ('(t + g)(x=x0)', Tensor),
    ('(g + g)(x=x0)', Tensor),
    ('(g + dy).reduce(ops.logaddexp, "x")', Joint),
    ('(g + dy).reduce(ops.logaddexp, "y")', Gaussian),
    ('(t + g + dy).reduce(ops.logaddexp, "x")', Joint),
    ('(t + g + dy).reduce(ops.logaddexp, "y")', Joint),
    ('(t + g).reduce(ops.logaddexp, "x")', Tensor),
]


@pytest.mark.parametrize('expr,expected_type', SMOKE_TESTS)
def test_smoke(expr, expected_type):
    dx = Delta('x', Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2))])))
    assert isinstance(dx, Delta)

    dy = Delta('y', Tensor(torch.randn(3, 4), OrderedDict([('j', bint(3))])))
    assert isinstance(dy, Delta)

    t = Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2)), ('j', bint(3))]))
    assert isinstance(t, Tensor)

    g = Gaussian(
        loc=torch.tensor([[0.0, 0.1, 0.2],
                          [2.0, 3.0, 4.0]]),
        precision=torch.tensor([[[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]],
                                [[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('x', reals(3))]))
    assert isinstance(g, Gaussian)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    result = eval(expr)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
    {'i': bint(2), 'j': bint(3)},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': reals()},
    {'x': reals(4)},
    {'x': reals(2, 3)},
    {'x': reals(), 'y': reals()},
    {'x': reals(2), 'y': reals(3)},
    {'x': reals(4), 'y': reals(2, 3), 'z': reals()},
], ids=id_from_inputs)
def test_reduce(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    t = random_tensor(int_inputs)
    g = random_gaussian(inputs)
    truth = {name: random_tensor(int_inputs, domain) for name, domain in real_inputs.items()}

    state = 0
    state += g
    state += t
    for name, point in truth.items():
        with xfail_if_not_implemented():
            state += Delta(name, point)
    actual = state.reduce(ops.logaddexp, frozenset(truth))

    expected = t + g(**truth)
    assert_close(actual, expected)
