# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from importlib import import_module

import numpy as np
import pytest

import funsor
import funsor.ops as ops
from funsor.distribution import BACKEND_TO_DISTRIBUTIONS_BACKEND
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.terms import Variable, lazy, to_data, to_funsor
from funsor.testing import assert_close, check_funsor, rand, randint, randn, random_scale_tril  # noqa: F401
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(get_backend() == "numpy",
                                reason="numpy does not have distributions backend")
if get_backend() != "numpy":
    dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    backend_dist = dist.dist

    class _fakes:
        def __getattribute__(self, attr):
            if get_backend() == "torch":
                return getattr(backend_dist.testing.fakes, attr)
            elif get_backend() == "jax":
                return getattr(dist, "_NumPyroWrapper_" + attr)
            raise ValueError(attr)

    FAKES = _fakes()


##################################################
# Test cases
##################################################

class DistTestCase:

    def __init__(self, raw_dist, raw_params, expected_value_domain):
        self.raw_dist = raw_dist
        self.raw_params = raw_params
        self.expected_value_domain = expected_value_domain
        for name, raw_param in self.raw_params:
            # we need direct access to these tensors for gradient tests
            setattr(self, name, eval(raw_param))

    def __str__(self):
        return self.raw_dist

    def __hash__(self):
        return hash((self.raw_dist, self.raw_params, self.expected_value_domain))


TEST_CASES = []

for batch_shape in [(), (5,), (2, 3)]:

    # Normal
    TEST_CASES += [DistTestCase(
        "backend_dist.Normal(case.loc, case.scale)",
        (("loc", f"randn({batch_shape})"), ("scale", f"rand({batch_shape})")),
        funsor.Real,
    )]
    # NonreparameterizedNormal
    TEST_CASES += [DistTestCase(
        "FAKES.NonreparameterizedNormal(case.loc, case.scale)",
        (("loc", f"randn({batch_shape})"), ("scale", f"rand({batch_shape})")),
        funsor.Real,
    )]

    # Beta
    TEST_CASES += [DistTestCase(
        "backend_dist.Beta(case.concentration1, case.concentration0)",
        (("concentration1", f"ops.exp(randn({batch_shape}))"), ("concentration0", f"ops.exp(randn({batch_shape}))")),
        funsor.Real,
    )]
    # NonreparameterizedBeta
    TEST_CASES += [DistTestCase(
        "FAKES.NonreparameterizedBeta(case.concentration1, case.concentration0)",
        (("concentration1", f"ops.exp(randn({batch_shape}))"), ("concentration0", f"ops.exp(randn({batch_shape}))")),
        funsor.Real,
    )]

    # Gamma
    TEST_CASES += [DistTestCase(
        "backend_dist.Gamma(case.concentration, case.rate)",
        (("concentration", f"rand({batch_shape})"), ("rate", f"rand({batch_shape})")),
        funsor.Real,
    )]
    # NonreparametrizedGamma
    TEST_CASES += [DistTestCase(
        "FAKES.NonreparameterizedGamma(case.concentration, case.rate)",
        (("concentration", f"rand({batch_shape})"), ("rate", f"rand({batch_shape})")),
        funsor.Real,
    )]

    # Dirichlet
    for event_shape in [(1,), (4,), (5,)]:
        TEST_CASES += [DistTestCase(
            "backend_dist.Dirichlet(case.concentration)",
            (("concentration", f"rand({batch_shape + event_shape})"),),
            funsor.Reals[event_shape],
        )]
        # NonreparameterizedDirichlet
        TEST_CASES += [DistTestCase(
            "FAKES.NonreparameterizedDirichlet(case.concentration)",
            (("concentration", f"rand({batch_shape + event_shape})"),),
            funsor.Reals[event_shape],
        )]

    # MultivariateNormal
    for event_shape in [(1,), (3,)]:
        TEST_CASES += [DistTestCase(
            "backend_dist.MultivariateNormal(case.loc, case.scale_tril)",
            (("loc", f"randn({batch_shape + event_shape})"), ("scale_tril", f"random_scale_tril({batch_shape + event_shape * 2})")),  # noqa: E501
            funsor.Reals[event_shape],
        )]

    # BernoulliLogits
    TEST_CASES += [DistTestCase(
        "backend_dist.Bernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        funsor.Real,
    )]


###########################
# Generic tests:
#   High-level distribution testing strategy: sequence of increasingly semantically strong distribution-agnostic tests
#   Conversion invertibility -> density type and value -> enumerate_support type and value -> samplers -> gradients
###########################

def _default_dim_to_name(inputs_shape, event_inputs=None):
    DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
    dim_to_name_list = DIM_TO_NAME + event_inputs if event_inputs else DIM_TO_NAME
    dim_to_name = OrderedDict(zip(
        range(-len(inputs_shape), 0),
        dim_to_name_list[len(dim_to_name_list) - len(inputs_shape):]))
    name_to_dim = OrderedDict((name, dim) for dim, name in dim_to_name.items())
    return dim_to_name, name_to_dim


def _get_stat(raw_dist, sample_shape, statistic):
    dim_to_name, name_to_dim = _default_dim_to_name(sample_shape + raw_dist.batch_shape)
    with interpretation(lazy):
        funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)

    sample_inputs = OrderedDict((dim_to_name[dim - len(raw_dist.batch_shape)], funsor.Bint[sample_shape[dim]])
                                for dim in range(-len(sample_shape), 0))
    rng_key = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    sample_value = funsor_dist.sample(frozenset(['value']), sample_inputs, rng_key=rng_key)
    expected_inputs = OrderedDict(tuple(sample_inputs.items()) + tuple(funsor_dist.inputs.items()))
    check_funsor(sample_value, expected_inputs, funsor.Real)

    expected_stat = getattr(funsor_dist, statistic)()
    if statistic == "mean":
        actual_stat = Integrate(
            sample_value, Variable('value', funsor_dist.inputs['value']), frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
    elif statistic == "variance":
        actual_mean = Integrate(
            sample_value, Variable('value', funsor_dist.inputs['value']), frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
        actual_stat = Integrate(
            sample_value,
            (Variable('value', funsor_dist.inputs['value']) - actual_mean) ** 2,
            frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
    elif statistic == "entropy":
        actual_stat = -Integrate(
            sample_value, funsor_dist, frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))
    else:
        raise ValueError("invalid test statistic: {}".format(statistic))

    return actual_stat, expected_stat


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
def test_generic_distribution_to_funsor(case):

    raw_dist, expected_value_domain = eval(case.raw_dist), case.expected_value_domain

    dim_to_name, name_to_dim = _default_dim_to_name(raw_dist.batch_shape)
    with interpretation(lazy):
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


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
def test_generic_log_prob(case):

    raw_dist, expected_value_domain = eval(case.raw_dist), case.expected_value_domain

    dim_to_name, name_to_dim = _default_dim_to_name(raw_dist.batch_shape)
    funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)
    expected_inputs = {name: funsor.Bint[raw_dist.batch_shape[dim]] for dim, name in dim_to_name.items()}
    expected_inputs.update({"value": expected_value_domain})

    check_funsor(funsor_dist, expected_inputs, funsor.Real)

    if get_backend() == "jax":
        raw_value = raw_dist.sample(key=np.array([0, 0], dtype=np.uint32))
    else:
        raw_value = raw_dist.sample()
    expected_logprob = to_funsor(raw_dist.log_prob(raw_value), output=funsor.Real, dim_to_name=dim_to_name)
    funsor_value = to_funsor(raw_value, output=expected_value_domain, dim_to_name=dim_to_name)
    assert_close(funsor_dist(value=funsor_value), expected_logprob)


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
@pytest.mark.parametrize("expand", [False, True])
def test_generic_enumerate_support(case, expand):

    raw_dist = eval(case.raw_dist)

    dim_to_name, name_to_dim = _default_dim_to_name(raw_dist.batch_shape)
    with interpretation(lazy):
        funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)

    assert getattr(raw_dist, "has_enumerate_support", False) == getattr(funsor_dist, "has_enumerate_support", False)
    if getattr(funsor_dist, "has_enumerate_support", False):
        name_to_dim["value"] = -1 if not name_to_dim else min(name_to_dim.values()) - 1
        raw_support = raw_dist.enumerate_support(expand=expand)
        funsor_support = funsor_dist.enumerate_support(expand=expand)
        assert_close(to_data(funsor_support, name_to_dim=name_to_dim), raw_support)


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
@pytest.mark.parametrize("statistic", ["mean", "variance", "entropy"])
@pytest.mark.parametrize("sample_shape", [(), (2,), (4, 3)], ids=str)
def test_generic_sample(case, statistic, sample_shape):

    raw_dist = eval(case.raw_dist)

    dim_to_name, name_to_dim = _default_dim_to_name(sample_shape + raw_dist.batch_shape)
    with interpretation(lazy):
        funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)

    sample_inputs = OrderedDict((dim_to_name[dim - len(raw_dist.batch_shape)], funsor.Bint[sample_shape[dim]])
                                for dim in range(-len(sample_shape), 0))
    rng_key = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    sample_value = funsor_dist.sample(frozenset(['value']), sample_inputs, rng_key=rng_key)
    expected_inputs = OrderedDict(tuple(sample_inputs.items()) + tuple(funsor_dist.inputs.items()))
    check_funsor(sample_value, expected_inputs, funsor.Real)


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
@pytest.mark.parametrize("statistic", ["mean", "variance", "entropy"])
@pytest.mark.parametrize("sample_shape", [(), (200000,), (400, 400)], ids=str)
def test_generic_stats(case, statistic, sample_shape):

    raw_dist = eval(case.raw_dist)

    atol = 1e-2

    actual_stat, expected_stat = _get_stat(raw_dist, sample_shape, statistic)
    check_funsor(actual_stat, expected_stat.inputs, expected_stat.output)
    assert_close(actual_stat.reduce(ops.add), expected_stat.reduce(ops.add), atol=atol, rtol=None)


@pytest.mark.skipif(get_backend() != "torch", reason="not working yet")
@pytest.mark.parametrize("case", TEST_CASES, ids=str)
@pytest.mark.parametrize("statistic", ["mean", "variance", "entropy"])
@pytest.mark.parametrize("sample_shape", [(200000,), (400, 400)], ids=str)
def test_generic_grads(case, statistic, sample_shape):

    raw_dist = eval(case.raw_dist)

    atol = 1e-2

    def _get_stat_diff_fn(raw_dist):
        actual_stat, expected_stat = _get_stat(raw_dist, sample_shape, statistic)
        return to_data((actual_stat - expected_stat).sum())

    if get_backend() == "torch":
        import torch

        params = tuple(getattr(case, param) for param, _ in case.raw_params)
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
