# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# High-level distribution testing strategy: sequence of increasingly semantically strong distribution-agnostic tests
# conversion invertibility -> density type and value -> enumerate_support type and value -> samplers -> gradients

from collections import OrderedDict, namedtuple
from importlib import import_module

import numpy as np
import pytest

import funsor
import funsor.ops as ops
from funsor.distribution import BACKEND_TO_DISTRIBUTIONS_BACKEND
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.terms import Variable, eager, lazy, to_data, to_funsor
from funsor.testing import assert_close, check_funsor, rand, randint, randn
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(get_backend() == "numpy",
                                reason="numpy does not have distributions backend")
if get_backend() != "numpy":
    dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    backend_dist = dist.dist


def _skip_for_numpyro_version(version="0.2.4"):
    if get_backend() == "jax":
        import numpyro

        if numpyro.__version__ <= version:
            return True

    return False


def default_dim_to_name(inputs_shape, event_inputs=None):
    DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
    dim_to_name_list = DIM_TO_NAME + event_inputs if event_inputs else DIM_TO_NAME
    dim_to_name = OrderedDict(zip(
        range(-len(inputs_shape), 0),
        dim_to_name_list[len(dim_to_name_list) - len(inputs_shape):]))
    name_to_dim = OrderedDict((name, dim) for dim, name in dim_to_name.items())
    return name_to_dim


def _get_stat(raw_dist, sample_shape, statistic, with_lazy):
    dim_to_name, name_to_dim = default_dim_to_name(sample_shape + raw_dist.batch_shape)
    with interpretation(lazy if with_lazy else eager):
        funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)

    sample_inputs = ...  # TODO compute sample_inputs from dim_to_name
    rng_key = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    sample_value = funsor_dist.sample(frozenset(['value']), sample_inputs, rng_key=rng_key)
    expected_inputs = OrderedDict(tuple(sample_inputs.items()) + tuple(funsor_dist.inputs.items()))
    check_funsor(sample_value, expected_inputs, funsor.Real)

    if statistic == "mean":
        actual_stat = Integrate(
            sample_value, Variable('value', funsor_dist.inputs['value']), frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
        expected_stat = funsor_dist.mean()
    elif statistic == "variance":
        actual_mean = Integrate(
            sample_value, Variable('value', funsor_dist.inputs['value']), frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
        actual_stat = Integrate(
            sample_value,
            (Variable('value', funsor_dist.inputs['value']) - actual_mean) ** 2,
            frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
        expected_stat = funsor_dist.variance()
    elif statistic == "entropy":
        actual_stat = -Integrate(
            sample_value, funsor_dist, frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
        expected_stat = funsor_dist.entropy()
    else:
        raise ValueError("invalid test statistic")

    return actual_stat.reduce(ops.add), expected_stat.reduce(ops.add)


##################################################
# Test cases
##################################################

# TODO how to make this work with multiple pytest workers?

DistTestCase = namedtuple("DistTestCase", ["raw_dist", "expected_value_domain"])

TEST_CASES = [
    DistTestCase(raw_dist=...),
]


###########################
# Generic tests
###########################

@pytest.mark.parametrize("case", TEST_CASES)
def test_generic_distribution_to_funsor(case):

    raw_dist, expected_value_domain = case.raw_dist, case.expected_value_domain

    dim_to_name, name_to_dim = default_dim_to_name(raw_dist.batch_shape)
    funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)
    actual_dist = to_data(funsor_dist, name_to_dim=name_to_dim)

    assert isinstance(actual_dist, backend_dist.Distribution)
    assert type(raw_dist) == type(actual_dist)
    assert funsor_dist.inputs["value"] == expected_value_domain
    for param_name in funsor_dist.params.keys():
        if param_name == "value":
            continue
        assert hasattr(raw_dist, param_name)
        assert_close(getattr(actual_dist, param_name), getattr(raw_dist, param_name))


@pytest.mark.parametrize("case", TEST_CASES)
def test_generic_log_prob(case):

    raw_dist, expected_value_domain = case.raw_dist, case.expected_value_domain

    dim_to_name, name_to_dim = default_dim_to_name(raw_dist.batch_shape)
    funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)
    expected_inputs = {name: funsor.Bint[raw_dist.batch_shape[dim]] for dim, name in dim_to_name.items()}
    expected_inputs.update({"value": expected_value_domain})

    check_funsor(funsor_dist, expected_inputs, funsor.Real)

    if get_backend() == "jax":
        raw_value = raw_dist.sample(rng_key=np.array([0, 0], dtype=np.uint32))
    else:
        raw_value = raw_dist.sample()
    expected_logprob = to_funsor(raw_dist.log_prob(raw_value), output=funsor.Real, dim_to_name=dim_to_name)
    funsor_value = to_funsor(raw_value, output=expected_value_domain, dim_to_name=dim_to_name)
    assert_close(funsor_dist(value=funsor_value), expected_logprob)


@pytest.mark.parametrize("case", TEST_CASES)
@pytest.mark.parametrize("expand", [False, True])
def test_generic_enumerate_support(case, expand):

    raw_dist = case.raw_dist

    dim_to_name, name_to_dim = default_dim_to_name(raw_dist.batch_shape)
    funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)

    assert getattr(raw_dist, "has_enumerate_support", False) == funsor_dist.has_enumerate_support
    if funsor_dist.has_enumerate_support:
        raw_support = raw_dist.enumerate_support(expand=expand)
        funsor_support = funsor_dist.enumerate_support(expand=expand)
        assert_close(to_data(funsor_support, name_to_dim=name_to_dim), raw_support)


@pytest.mark.parametrize("case", TEST_CASES)
@pytest.mark.parametrize("statistic", ["mean", "variance", "entropy"])
@pytest.mark.parametrize("with_lazy", [True, False])
def test_generic_sample(case, statistic, with_lazy):

    raw_dist, sample_shape = case.raw_dist, case.sample_shape

    atol = 1e-2

    actual_stat, expected_stat = _get_stat(raw_dist, sample_shape, statistic, with_lazy)
    check_funsor(actual_stat, expected_stat.inputs, expected_stat.output)
    if sample_shape:
        assert_close(actual_stat, expected_stat, atol=atol, rtol=None)


@pytest.mark.skipif(True, reason="broken")
@pytest.mark.parametrize("case", TEST_CASES)
@pytest.mark.parametrize("statistic", ["mean", "variance", "entropy"])
@pytest.mark.parametrize("with_lazy", [True, False])
def test_generic_sample_grads(case, statistic, with_lazy):

    raw_dist, sample_shape = case.raw_dist, case.sample_shape

    atol = 1e-2

    def _get_stat_diff_fn(raw_dist):
        actual_stat, expected_stat = _get_stat(raw_dist, sample_shape, statistic, with_lazy)
        return to_data((actual_stat - expected_stat).sum())

    if get_backend() == "torch":
        import torch

        # TODO compute params here
        for param in params:
            param.requires_grad_()

        diff = _get_stat_diff_fn(raw_dist)
        assert_close(diff, ops.new_zeros(diff, diff.shape), atol=atol, rtol=None)
        diff_grads = torch.autograd.grad(diff, params, allow_unused=True)
        for diff_grad in diff_grads:
            assert_close(diff_grad, ops.new_zeros(diff_grad, diff_grad.shape), atol=atol, rtol=None)

    elif get_backend() == "jax":
        import jax

        # TODO compute gradient wrt distribution instance PyTree
        diff, diff_grads = jax.value_and_grad(lambda *args: _get_stat_diff_fn(*args).sum(), has_aux=True)(params)
        assert_close(diff, ops.new_zeros(diff, diff.shape), atol=atol, rtol=None)
        for diff_grad in diff_grads:
            assert_close(diff_grad, ops.new_zeros(diff_grad, diff_grad.shape), atol=atol, rtol=None)
