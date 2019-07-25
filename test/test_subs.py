from collections import OrderedDict

import pytest
import torch

import funsor.distributions as dist
from funsor.delta import Delta
from funsor.domains import bint
from funsor.interpreter import interpretation
from funsor.terms import Number, lazy, normalize_simultaneous_subs
from funsor.torch import Tensor


# TODO add more test cases
TEST_CASES = [
    OrderedDict({"a": "t", "b": "g"}),
    OrderedDict({"a": "t", "b": "g + t"}),
    OrderedDict({"a": "t", "b": "dx", "c": "dx + dy"}),
    OrderedDict({"i": "i0", "b": "t"}),
    OrderedDict({"i": "i0", "b": "dx"}),
    OrderedDict({"i": "i0", "b": "dx + t"}),
    OrderedDict({"x": "t", "i": "i0", "b": "g"}),
    OrderedDict({"x": "t", "j": "j0", "i": "i0", "b": "g"}),
]


@pytest.mark.parametrize("exprs", TEST_CASES)
def test_simultaneous_subs(exprs):

    dx = Delta('x', Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2))])))
    assert isinstance(dx, Delta)

    dy = Delta('y', Tensor(torch.randn(3, 4), OrderedDict([('j', bint(3))])))
    assert isinstance(dy, Delta)

    t = Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2)), ('j', bint(3))]))
    assert isinstance(t, Tensor)

    g = dist.Normal(loc=0, scale=1, value='x')

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    j0 = Number(1, 3)
    assert isinstance(j0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    ctx = locals()
    with interpretation(lazy):
        old_exprs = exprs
        exprs = OrderedDict((k, eval(v, globals(), ctx)) for k, v in exprs.items())

    res = normalize_simultaneous_subs(exprs)
    assert isinstance(res, tuple)

    expected_outputs = frozenset(exprs.keys())
    actual_outputs = frozenset(k for k, v in res)
    assert actual_outputs == expected_outputs

    expected_inputs = frozenset().union(*(e.inputs for e in exprs.values())) - frozenset(exprs.keys())
    actual_inputs = OrderedDict()
    for k, v in res:
        actual_inputs.update(v.inputs)
    assert frozenset(actual_inputs) == expected_inputs

    assert not any(actual_outputs.intersection(outp.inputs)
                   for k, outp in res)
