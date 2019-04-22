from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch
from six.moves import reduce

import funsor.distributions as dist
import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.interpreter import interpretation
from funsor.joint import Joint
from funsor.product import Product
from funsor.terms import Number, Reduce, Subs, lazy
from funsor.testing import assert_close, random_gaussian, random_tensor, xfail_if_not_implemented
from funsor.torch import Tensor


# TODO add more test cases
TEST_CASES = [
    ({"a": "t", "b": "g"}, Product),
    ({"a": "t", "b": "g + t"}, Product),
    ({"a": "t", "b": "dx", "c": "dx + dy"}, Product),
    ({"i": "i0", "b": "t"}, Product),
    ({"i": "i0", "b": "dx"}, Product),
    ({"i": "i0", "b": "dx + t"}, Product),
    ({"x": "t", "i": "i0", "b": "g"}, Product),
    ({"x": "t", "j": "j0", "i": "i0", "b": "g"}, Product),
]


@pytest.mark.parametrize("exprs,expected_type", TEST_CASES)
def test_product_normal_form(exprs, expected_type):

    dx = Delta('x', Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2))])))
    assert isinstance(dx, Delta)

    dy = Delta('y', Tensor(torch.randn(3, 4), OrderedDict([('j', bint(3))])))
    assert isinstance(dy, Delta)

    t = Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2)), ('j', bint(3))]))
    assert isinstance(t, Tensor)

    g = dist.Normal(loc=0, scale=1, value='x')
    assert isinstance(g, Joint)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    j0 = Number(1, 2)
    assert isinstance(j0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    ctx = locals()
    with interpretation(lazy):
        exprs = {k: eval(v, globals(), ctx) for k, v in exprs.items()}

    res = Product(exprs)
    assert isinstance(res, expected_type)

    expected_outputs = frozenset(exprs.keys())
    assert frozenset(res.outputs) == expected_outputs

    expected_inputs = frozenset().union(*(e.inputs for e in exprs.values())) - frozenset(exprs.keys())
    assert frozenset(res.inputs) == expected_inputs
        
    assert not any(frozenset(res.outputs.keys()).intersection(outp.inputs)
                   for outp in res.outputs.values())

    assert all(isinstance(value, (Subs, type(exprs[name]), Tensor, Joint))
               for name, value in res.outputs.items())
