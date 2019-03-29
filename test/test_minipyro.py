from __future__ import absolute_import, division, print_function

import warnings

import pytest
import torch
from pyro.generic import distributions as dist
from pyro.generic import infer, optim, pyro, pyro_backend

from funsor.testing import xfail_param

# This file tests a variety of model,guide pairs with valid and invalid structure.
# See https://github.com/pyro-ppl/pyro/blob/0.3.1/tests/infer/test_valid_models.py


def assert_ok(model, guide, elbo, *args, **kwargs):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.get_param_store().clear()
    adam = optim.Adam({"lr": 1e-6})
    inference = infer.SVI(model, guide, adam, elbo)
    inference.step(*args, **kwargs)


def assert_error(model, guide, elbo, match=None):
    """
    Assert that inference fails with an error.
    """
    pyro.get_param_store().clear()
    adam = optim.Adam({"lr": 1e-6})
    inference = infer.SVI(model,  guide, adam, elbo)
    with pytest.raises((NotImplementedError, UserWarning, KeyError, ValueError, RuntimeError),
                       match=match):
        inference.step()


def assert_warning(model, guide, elbo):
    """
    Assert that inference works but with a warning.
    """
    pyro.get_param_store().clear()
    adam = optim.Adam({"lr": 1e-6})
    inference = infer.SVI(model, guide, adam, elbo)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inference.step()
        assert len(w), 'No warnings were raised'
        for warning in w:
            print(warning)


@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_nonempty_model_empty_guide_ok(backend):

    def model(data):
        loc = pyro.param("loc", torch.tensor(0.0))
        pyro.sample("x", dist.Normal(loc, 1.), obs=data)

    def guide(data):
        pass

    data = torch.tensor(2.)
    with pyro_backend(backend):
        elbo = infer.Trace_ELBO()
        assert_ok(model, guide, elbo, data)


@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_plate_ok(backend):
    data = torch.randn(10)

    def model():
        locs = pyro.param("locs", torch.tensor([0.2, 0.3, 0.5]))
        p = torch.tensor([0.2, 0.3, 0.5])
        with pyro.plate("plate", len(data), dim=-1):
            x = pyro.sample("x", dist.Categorical(p))
            pyro.sample("obs", dist.Normal(locs[x], 1.), obs=data)

    def guide():
        p = pyro.param("p", torch.tensor([0.5, 0.3, 0.2]))
        with pyro.plate("plate", len(data), dim=-1):
            pyro.sample("x", dist.Categorical(p))

    with pyro_backend(backend):
        elbo = infer.Trace_ELBO()
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("backend", [
    "pyro",
    xfail_param("funsor", reason="missing patterns"),
])
def test_mean_field_ok(backend):

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.Normal(x, 1.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        x = pyro.sample("x", dist.Normal(loc, 1.))
        pyro.sample("y", dist.Normal(x, 1.))

    with pyro_backend(backend):
        elbo = infer.TraceMeanField_ELBO()
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("backend", [
    "pyro",
    xfail_param("funsor", reason="missing patterns"),
])
def test_mean_field_warn(backend):

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.Normal(x, 1.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        y = pyro.sample("y", dist.Normal(loc, 1.))
        pyro.sample("x", dist.Normal(y, 1.))

    with pyro_backend(backend):
        elbo = infer.TraceMeanField_ELBO()
        assert_warning(model, guide, elbo)
