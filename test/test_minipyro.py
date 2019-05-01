from __future__ import absolute_import, division, print_function

import warnings

import pytest
import torch

from torch.autograd import grad
from torch.distributions import constraints, kl_divergence

from pyro.generic import distributions as dist
from pyro.generic import infer, optim, pyro, pyro_backend

import funsor
from funsor.testing import assert_close, xfail_param

# This file tests a variety of model,guide pairs with valid and invalid structure.
# See https://github.com/pyro-ppl/pyro/blob/0.3.1/tests/infer/test_valid_models.py


def assert_ok(model, guide, elbo, *args, **kwargs):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.get_param_store().clear()
    adam = optim.Adam({"lr": 1e-6})
    inference = infer.SVI(model, guide, adam, elbo)
    for i in range(2):
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
def test_generate_data(backend):

    def model(data=None):
        loc = pyro.param("loc", torch.tensor(2.0))
        scale = pyro.param("scale", torch.tensor(1.0))
        x = pyro.sample("x", dist.Normal(loc, scale), obs=data)
        return x

    with pyro_backend(backend):
        data = model().data
        assert data.shape == ()


@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_generate_data_plate(backend):
    num_points = 1000

    def model(data=None):
        loc = pyro.param("loc", torch.tensor(2.0))
        scale = pyro.param("scale", torch.tensor(1.0))
        with pyro.plate("data", 1000, dim=-1):
            x = pyro.sample("x", dist.Normal(loc, scale), obs=data)
        return x

    with pyro_backend(backend):
        data = model().data
        assert data.shape == (num_points,)
        mean = data.sum().item() / num_points
        assert 1.9 <= mean <= 2.1


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_nonempty_model_empty_guide_ok(backend, jit):

    def model(data):
        loc = pyro.param("loc", torch.tensor(0.0))
        pyro.sample("x", dist.Normal(loc, 1.), obs=data)

    def guide(data):
        pass

    data = torch.tensor(2.)
    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo, data)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_plate_ok(backend, jit):
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
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_nested_plate_plate_ok(backend, jit):
    data = torch.randn(2, 3)

    def model():
        loc = torch.tensor(3.0)
        with pyro.plate("plate_outer", data.size(-1), dim=-1):
            x = pyro.sample("x", dist.Normal(loc, 1.))
            with pyro.plate("plate_inner", data.size(-2), dim=-2):
                pyro.sample("y", dist.Normal(x, 1.), obs=data)

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        scale = pyro.param("scale", torch.tensor(1.))
        with pyro.plate("plate_outer", data.size(-1), dim=-1):
            pyro.sample("x", dist.Normal(loc, scale))

    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "funsor"])
def test_local_param_ok(backend, jit):
    data = torch.randn(10)

    def model():
        locs = pyro.param("locs", torch.tensor([-1., 0., 1.]))
        with pyro.plate("plate", len(data), dim=-1):
            x = pyro.sample("x", dist.Categorical(torch.ones(3) / 3))
            pyro.sample("obs", dist.Normal(locs[x], 1.), obs=data)

    def guide():
        with pyro.plate("plate", len(data), dim=-1):
            p = pyro.param("p", torch.ones(len(data), 3) / 3, event_dim=1)
            pyro.sample("x", dist.Categorical(p))
        return p

    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)

        # Check that pyro.param() can be called without init_value.
        expected = guide()
        actual = pyro.param("p")
        assert_close(actual, expected)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_constraints(backend, jit):
    data = torch.tensor(0.5)

    def model():
        locs = pyro.param("locs", torch.randn(3), constraint=constraints.real)
        scales = pyro.param("scales", torch.randn(3).exp(), constraint=constraints.positive)
        p = torch.tensor([0.5, 0.3, 0.2])
        x = pyro.sample("x", dist.Categorical(p))
        pyro.sample("obs", dist.Normal(locs[x], scales[x]), obs=data)

    def guide():
        q = pyro.param("q", torch.randn(3).exp(), constraint=constraints.simplex)
        pyro.sample("x", dist.Categorical(q))

    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
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


@pytest.mark.parametrize("backend", ["pyro", "funsor"])
@pytest.mark.parametrize("enumerate4", ["parallel"])
@pytest.mark.parametrize("enumerate3", ["parallel"])
@pytest.mark.parametrize("enumerate2", ["parallel"])
@pytest.mark.parametrize("enumerate1", ["parallel"])
@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
def test_elbo_plate_plate(backend, outer_dim, inner_dim, enumerate1, enumerate2, enumerate3, enumerate4):
    with pyro_backend(backend):
        pyro.clear_param_store()
        num_particles = 1 if all([enumerate1, enumerate2, enumerate3, enumerate4]) else 100000
        q = pyro.param("q", torch.tensor([0.75, 0.25], requires_grad=True))
        p = 0.2693204236205713  # for which kl(Categorical(q), Categorical(p)) = 0.5
        p = torch.tensor([p, 1-p])

        def model():
            d = dist.Categorical(p)
            context1 = pyro.plate("outer", outer_dim, dim=-1)
            context2 = pyro.plate("inner", inner_dim, dim=-2)
            pyro.sample("w", d)
            with context1:
                pyro.sample("x", d)
            with context2:
                pyro.sample("y", d)
            with context1, context2:
                pyro.sample("z", d)

        def guide():
            d = dist.Categorical(pyro.param("q"))
            context1 = pyro.plate("outer", outer_dim, dim=-1)
            context2 = pyro.plate("inner", inner_dim, dim=-2)
            pyro.sample("w", d, infer={"enumerate": enumerate1})
            with context1:
                pyro.sample("x", d, infer={"enumerate": enumerate2})
            with context2:
                pyro.sample("y", d, infer={"enumerate": enumerate3})
            with context1, context2:
                pyro.sample("z", d, infer={"enumerate": enumerate4})

        kl_node = kl_divergence(torch.distributions.Categorical(funsor.to_data(q)),
                                torch.distributions.Categorical(funsor.to_data(p)))
        kl = (1 + outer_dim + inner_dim + outer_dim * inner_dim) * kl_node
        expected_loss = kl
        expected_grad = grad(kl, [funsor.to_data(q)])[0]

        elbo = infer.TraceEnum_ELBO(num_particles=num_particles,
                                    vectorize_particles=True,
                                    strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
        elbo = elbo.differentiable_loss if backend == "pyro" else elbo
        actual_loss = funsor.to_data(elbo(model, guide))
        actual_loss.backward()
        actual_grad = funsor.to_data(pyro.param('q')).grad

        assert_close(actual_loss, expected_loss, atol=1e-5)
        assert_close(actual_grad, expected_grad, atol=1e-5)
