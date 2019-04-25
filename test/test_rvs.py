from __future__ import absolute_import, division, print_function

import math
import pytest
import torch
from collections import OrderedDict

import funsor.distributions as dist
import funsor.ops as ops
from funsor.domains import reals
from funsor.rvs import RandomVariable, BernoulliRV, NormalRV, CategoricalRV, DeltaRV
from funsor.terms import Funsor, Number, Variable
from funsor.testing import assert_close
from funsor.torch import Tensor


SMOKE_TEST_CASES = [
    ("NormalRV(0, 1)", NormalRV),
    ("CategoricalRV(Tensor(torch.ones(3) / 3, OrderedDict([])))", CategoricalRV),
    ("DeltaRV(5., 0)", DeltaRV),
    ("BernoulliRV(0.5)", BernoulliRV),
]


@pytest.mark.parametrize("expr,expected_type", SMOKE_TEST_CASES)
def test_sample_rv_smoke(expr, expected_type):
    x = eval(expr)
    assert isinstance(x, expected_type)
    assert isinstance(x.sample(frozenset(['omega'])), Tensor)


@pytest.mark.parametrize("expr,expected_type", SMOKE_TEST_CASES)
def test_mean_rv_smoke(expr, expected_type):
    x = eval(expr)
    assert isinstance(x, expected_type)
    assert isinstance(x.reduce(ops.add, frozenset(['omega'])), Tensor)


@pytest.mark.parametrize("expr,expected_type", SMOKE_TEST_CASES)
def test_subs_omega_ok_rv_smoke(expr, expected_type):
    x = eval(expr)
    assert isinstance(x, expected_type)
    # it's ok to substitute another variable of the same type into omega
    y = x(omega=Variable('omega2', reals()))
    assert isinstance(y, expected_type)


@pytest.mark.parametrize("expr,expected_type", SMOKE_TEST_CASES)
def test_subs_omega_fail_rv_smoke(expr, expected_type):
    x = eval(expr)
    assert isinstance(x, expected_type)
    # it's not ok to substitute an arbitrary expression into omega, no matter the type
    with pytest.raises(ValueError):
        z = x(omega=Number(1.5))
        assert isinstance(z, Funsor)


def test_transform_normalrv_simple():
    x = NormalRV(loc=0., scale=1.)
    y_actual = 2 * x + 1.

    y_expected = NormalRV(loc=1., scale=2.)

    actual = y_actual.reduce(ops.add)
    expected = y_expected.reduce(ops.add)

    assert_close(actual, expected)


def test_reduce_normalrv_simple():
    # check that reducing substitution is the same as reducing joint densities
    x = NormalRV(loc=0., scale=1., omega="omega_x")
    y = dist.Normal(loc=x, scale=1., value=0.5)
    actual = y.reduce(ops.logaddexp, "omega_x")

    expected = (dist.Normal(loc=0., scale=1., value="x") + dist.Normal(loc="x", scale=1., value=0.5)).reduce(ops.logaddexp)

    assert_close(actual, expected)


def test_add_normalrv_simple():
    x = NormalRV(loc=0., scale=1.)(omega='omegax')
    y = NormalRV(loc=1., scale=0.5)(omega='omegay')
    actual = x + y

    expected = NormalRV(loc=0. + 1., scale=math.sqrt(1.25))

    actual_mean = actual.reduce(ops.add)
    expected_mean = expected.reduce(ops.add)

    assert_close(actual_mean, expected_mean)


def test_add_normalrv_name_collision():
    x = NormalRV(loc=0., scale=1.)(omega='omega2')
    y = NormalRV(loc=1., scale=0.5)(omega='omega2')

    # this should fail because the omegas are the same and we don't like that
    with pytest.raises(NotImplementedError):
        x + y
