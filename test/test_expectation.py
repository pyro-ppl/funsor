from __future__ import absolute_import, division, print_function

import pytest
import torch
from collections import OrderedDict
from six.moves import reduce

import funsor.distributions as dist
import funsor.ops as ops
from funsor.domains import bint
from funsor.expectation import Expectation
from funsor.joint import Joint
from funsor.product import Product
from funsor.rvs import RandomVariable, BernoulliRV, NormalRV, CategoricalRV, DeltaRV
from funsor.testing import assert_close
from funsor.torch import Tensor


@pytest.mark.parametrize("measure,integrand", [
    (Product({"x": NormalRV(0., 1.)}), dist.Normal('x', 1., 0.5)),
    (Product({"x": CategoricalRV(Tensor(torch.ones(3) / 3., OrderedDict([])))}),
     dist.Categorical(Tensor(torch.ones(3, 3) / 3., OrderedDict([('x', bint(3))])), 1)),
    (Product({"x": DeltaRV(1.)}), dist.Normal('x', 1., 0.5)),
])
def test_expectation_single_rv(measure, integrand):
    # this tests expectation against Joint and Integrate, basically
    assert isinstance(measure, Product)

    joint = reduce(ops.add, [v.log_prob(k) for k, v in measure.outputs.items()])

    # handle missing patterns...
    if isinstance(joint, Joint) and isinstance(integrand, Joint):
        with pytest.raises(NotImplementedError):
            expected = (joint.exp() * integrand).reduce(ops.add, frozenset(measure.outputs))
    else:
        expected = (joint.exp() * integrand).reduce(ops.add, frozenset(measure.outputs))

        actual = Expectation(measure, integrand)

        assert isinstance(actual, Tensor)
        assert isinstance(expected, Tensor)
        assert_close(actual, expected)
