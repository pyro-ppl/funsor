# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import OrderedDict
from importlib import import_module

import numpy as np
import pytest

import funsor
import funsor.ops as ops
from funsor.distribution import BACKEND_TO_DISTRIBUTIONS_BACKEND
from funsor.interpreter import interpretation
from funsor.terms import lazy, to_data, to_funsor
from funsor.testing import assert_close, check_funsor, rand, randint, randn, random_scale_tril, xfail_if_not_implemented  # noqa: F401,E501
from funsor.util import get_backend


_ENABLE_MC_DIST_TESTS = int(os.environ.get("FUNSOR_ENABLE_MC_DIST_TESTS", 0))

pytestmark = pytest.mark.skipif(get_backend() == "numpy",
                                reason="numpy does not have distributions backend")
if get_backend() != "numpy":
    dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    backend_dist = dist.dist

    class _fakes:
        """alias for accessing nonreparameterized distributions"""
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

TEST_CASES = []


class DistTestCase:

    def __init__(self, raw_dist, raw_params, expected_value_domain):
        self.raw_dist = raw_dist
        self.raw_params = raw_params
        self.expected_value_domain = expected_value_domain
        for name, raw_param in self.raw_params:
            if get_backend() != "numpy":
                # we need direct access to these tensors for gradient tests
                setattr(self, name, eval(raw_param))
        TEST_CASES.append(self)

    def __str__(self):
        return self.raw_dist + " " + str(self.raw_params)

    def __hash__(self):
        return hash((self.raw_dist, self.raw_params, self.expected_value_domain))


for batch_shape in [(), (5,), (2, 3)]:

    # BernoulliLogits
    DistTestCase(
        "backend_dist.Bernoulli(logits=case.logits)",
        (("logits", f"rand({batch_shape})"),),
        funsor.Real,
    )

    # BernoulliProbs
    DistTestCase(
        "backend_dist.Bernoulli(probs=case.probs)",
        (("probs", f"rand({batch_shape})"),),
        funsor.Real,
    )

    # Beta
    DistTestCase(
        "backend_dist.Beta(case.concentration1, case.concentration0)",
        (("concentration1", f"ops.exp(randn({batch_shape}))"), ("concentration0", f"ops.exp(randn({batch_shape}))")),
        funsor.Real,
    )
    # NonreparameterizedBeta
    DistTestCase(
        "FAKES.NonreparameterizedBeta(case.concentration1, case.concentration0)",
        (("concentration1", f"ops.exp(randn({batch_shape}))"), ("concentration0", f"ops.exp(randn({batch_shape}))")),
        funsor.Real,
    )

    # Binomial
    DistTestCase(
        "backend_dist.Binomial(total_count=case.total_count, probs=case.probs)",
        (("total_count", "randint(10, 12, ())" if get_backend() == "jax" else "5"), ("probs", f"rand({batch_shape})")),
        funsor.Real,
    )

    # CategoricalLogits
    for size in [2, 4]:
        DistTestCase(
            "backend_dist.Categorical(logits=case.logits)",
            (("logits", f"rand({batch_shape + (size,)})"),),
            funsor.Bint[size],
        )

    # CategoricalProbs
    for size in [2, 4]:
        DistTestCase(
            "backend_dist.Categorical(probs=case.probs)",
            (("probs", f"rand({batch_shape + (size,)})"),),
            funsor.Bint[size],
        )

    # TODO figure out what this should be...
    # # Delta
    # for event_shape in [(),]:  # (4,), (3, 2)]:
    #     TEST_CASES += [DistTestCase(
    #         "backend_dist.Delta(case.v, case.log_density)",
    #         (("v", f"rand({batch_shape + event_shape})"), ("log_density", f"rand({batch_shape})")),
    #         funsor.Real,  # s[event_shape],
    #     )]

    # Dirichlet
    for event_shape in [(1,), (4,)]:
        DistTestCase(
            "backend_dist.Dirichlet(case.concentration)",
            (("concentration", f"rand({batch_shape + event_shape})"),),
            funsor.Reals[event_shape],
        )
        # NonreparameterizedDirichlet
        DistTestCase(
            "FAKES.NonreparameterizedDirichlet(case.concentration)",
            (("concentration", f"rand({batch_shape + event_shape})"),),
            funsor.Reals[event_shape],
        )

    # DirichletMultinomial
    for event_shape in [(1,), (4,)]:
        DistTestCase(
            "backend_dist.DirichletMultinomial(case.concentration, case.total_count)",
            (("concentration", f"rand({batch_shape + event_shape})"), ("total_count", "randint(10, 12, ())")),
            funsor.Reals[event_shape],
        )

    # Gamma
    DistTestCase(
        "backend_dist.Gamma(case.concentration, case.rate)",
        (("concentration", f"rand({batch_shape})"), ("rate", f"rand({batch_shape})")),
        funsor.Real,
    )
    # NonreparametrizedGamma
    DistTestCase(
        "FAKES.NonreparameterizedGamma(case.concentration, case.rate)",
        (("concentration", f"rand({batch_shape})"), ("rate", f"rand({batch_shape})")),
        funsor.Real,
    )

    # Multinomial
    for event_shape in [(1,), (4,)]:
        DistTestCase(
            "backend_dist.Multinomial(case.total_count, probs=case.probs)",
            (("total_count", "randint(5, 7, ())" if get_backend() == "jax" else "5"),
             ("probs", f"rand({batch_shape + event_shape})")),
            funsor.Reals[event_shape],
        )

    # MultivariateNormal
    for event_shape in [(1,), (3,)]:
        DistTestCase(
            "backend_dist.MultivariateNormal(loc=case.loc, scale_tril=case.scale_tril)",
            (("loc", f"randn({batch_shape + event_shape})"), ("scale_tril", f"random_scale_tril({batch_shape + event_shape * 2})")),  # noqa: E501
            funsor.Reals[event_shape],
        )

    # Normal
    DistTestCase(
        "backend_dist.Normal(case.loc, case.scale)",
        (("loc", f"randn({batch_shape})"), ("scale", f"rand({batch_shape})")),
        funsor.Real,
    )
    # NonreparameterizedNormal
    DistTestCase(
        "FAKES.NonreparameterizedNormal(case.loc, case.scale)",
        (("loc", f"randn({batch_shape})"), ("scale", f"rand({batch_shape})")),
        funsor.Real,
    )

    # Poisson
    DistTestCase(
        "backend_dist.Poisson(rate=case.rate)",
        (("rate", f"rand({batch_shape})"),),
        funsor.Real,
    )

    # VonMises
    DistTestCase(
        "backend_dist.VonMises(case.loc, case.concentration)",
        (("loc", f"rand({batch_shape})"), ("concentration", f"rand({batch_shape})")),
        funsor.Real,
    )

    # Independent
    for indep_shape in [(3,), (2, 3)]:
        # Beta.to_event
        DistTestCase(
            f"backend_dist.Beta(case.concentration1, case.concentration0).to_event({len(indep_shape)})",
            (("concentration1", f"ops.exp(randn({batch_shape + indep_shape}))"),
             ("concentration0", f"ops.exp(randn({batch_shape + indep_shape}))")),
            funsor.Reals[indep_shape],
        )
        # Dirichlet.to_event
        for event_shape in [(2,), (4,)]:
            DistTestCase(
                f"backend_dist.Dirichlet(case.concentration).to_event({len(indep_shape)})",
                (("concentration", f"rand({batch_shape + indep_shape + event_shape})"),),
                funsor.Reals[indep_shape + event_shape],
            )


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


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
def test_generic_distribution_to_funsor(case):

    raw_dist, expected_value_domain = eval(case.raw_dist), case.expected_value_domain

    dim_to_name, name_to_dim = _default_dim_to_name(raw_dist.batch_shape)
    with interpretation(lazy):
        funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)
    assert funsor_dist.inputs["value"] == expected_value_domain

    actual_dist = to_data(funsor_dist, name_to_dim=name_to_dim)

    assert isinstance(actual_dist, backend_dist.Distribution)
    assert issubclass(type(actual_dist), type(raw_dist))  # subclass to handle wrappers
    while isinstance(raw_dist, (backend_dist.Independent, backend_dist.TransformedDistribution)):
        raw_dist = raw_dist.base_dist
        actual_dist = actual_dist.base_dist
        assert isinstance(actual_dist, backend_dist.Distribution)
        assert issubclass(type(actual_dist), type(raw_dist))  # subclass to handle wrappers

    for param_name, _ in case.raw_params:
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
    assert_close(funsor_dist(value=funsor_value), expected_logprob, rtol=5e-4)


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
        with xfail_if_not_implemented("enumerate support not implemented"):
            raw_support = raw_dist.enumerate_support(expand=expand)
            funsor_support = funsor_dist.enumerate_support(expand=expand)
            assert_close(to_data(funsor_support, name_to_dim=name_to_dim), raw_support)


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
@pytest.mark.parametrize("sample_shape", [(), (2,), (4, 3)], ids=str)
def test_generic_sample(case, sample_shape):

    raw_dist = eval(case.raw_dist)

    dim_to_name, name_to_dim = _default_dim_to_name(sample_shape + raw_dist.batch_shape)
    with interpretation(lazy):
        funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)

    sample_inputs = OrderedDict((dim_to_name[dim - len(raw_dist.batch_shape)], funsor.Bint[sample_shape[dim]])
                                for dim in range(-len(sample_shape), 0))
    rng_key = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    sample_value = funsor_dist.sample(frozenset(['value']), sample_inputs, rng_key=rng_key)
    expected_inputs = OrderedDict(tuple(sample_inputs.items()) + tuple(funsor_dist.inputs.items()))
    # TODO compare sample values on jax backend
    check_funsor(sample_value, expected_inputs, funsor.Real)


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
@pytest.mark.parametrize("statistic", [
    "mean",
    "variance",
    pytest.param("entropy", marks=[pytest.mark.skipif(get_backend() == "jax", reason="entropy not implemented")])
])
def test_generic_stats(case, statistic):

    raw_dist = eval(case.raw_dist)

    dim_to_name, name_to_dim = _default_dim_to_name(raw_dist.batch_shape)
    with interpretation(lazy):
        funsor_dist = to_funsor(raw_dist, output=funsor.Real, dim_to_name=dim_to_name)

    with xfail_if_not_implemented(msg="entropy not implemented for some distributions"):
        actual_stat = getattr(funsor_dist, statistic)()

    expected_stat_raw = getattr(raw_dist, statistic)
    if statistic == "entropy":
        expected_stat = to_funsor(expected_stat_raw(), output=funsor.Real, dim_to_name=dim_to_name)
    else:
        expected_stat = to_funsor(expected_stat_raw, output=case.expected_value_domain, dim_to_name=dim_to_name)

    check_funsor(actual_stat, expected_stat.inputs, expected_stat.output)
    if ops.isnan(expected_stat.data).all():
        pytest.xfail(reason="base stat returns nan")
    else:
        assert_close(to_data(actual_stat, name_to_dim), to_data(expected_stat, name_to_dim), rtol=1e-4)
