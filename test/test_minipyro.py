# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

import funsor
from funsor.testing import xfail_param
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "torch",
                                reason="numpy/jax backend requires porting pyro.ops.einsum")
if get_backend() == "torch":
    import torch
    from pyro.ops.indexing import Vindex as _Vindex
    from pyroapi import distributions as dist
    from pyroapi import handlers, infer, optim, pyro, pyro_backend
    from torch.autograd import grad
    from torch.distributions import constraints, kl_divergence

    import funsor.compat.ops as ops


# This file tests a variety of model,guide pairs with valid and invalid structure.
# See https://github.com/pyro-ppl/pyro/blob/0.3.1/tests/infer/test_valid_models.py


def Vindex(x):
    if isinstance(x, funsor.Funsor):
        return x
    return _Vindex(x)


def _check_loss_and_grads(expected_loss, actual_loss, atol=1e-4, rtol=1e-4):
    # copied from pyro
    expected_loss, actual_loss = funsor.to_data(expected_loss), funsor.to_data(actual_loss)
    assert ops.allclose(actual_loss, expected_loss, atol=atol, rtol=rtol)
    names = pyro.get_param_store().keys()
    params = []
    for name in names:
        params.append(funsor.to_data(pyro.param(name)).unconstrained())
    actual_grads = grad(actual_loss, params, allow_unused=True, retain_graph=True)
    expected_grads = grad(expected_loss, params, allow_unused=True, retain_graph=True)
    for name, actual_grad, expected_grad in zip(names, actual_grads, expected_grads):
        if actual_grad is None or expected_grad is None:
            continue
        assert ops.allclose(actual_grad, expected_grad, atol=atol, rtol=rtol)


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

    def model():
        loc = pyro.param("loc", torch.tensor(2.0))
        scale = pyro.param("scale", torch.tensor(1.0))
        x = pyro.sample("x", dist.Normal(loc, scale))
        return x

    with pyro_backend(backend):
        data = model()
        data = data.data
        assert data.shape == ()


@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_rng_seed(backend):

    def model():
        return pyro.sample("x", dist.Normal(0, 1))

    with pyro_backend(backend):
        with handlers.seed(rng_seed=0):
            expected = model()
        with handlers.seed(rng_seed=0):
            actual = model()
        assert ops.allclose(actual, expected)


@pytest.mark.parametrize("backend", ["pyro", "minipyro", "funsor"])
def test_rng_state(backend):

    def model():
        return pyro.sample("x", dist.Normal(0, 1))

    with pyro_backend(backend):
        with handlers.seed(rng_seed=0):
            model()
            expected = model()
        with handlers.seed(rng_seed=0):
            model()
            with handlers.seed(rng_seed=0):
                model()
            actual = model()
        assert ops.allclose(actual, expected)


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
@pytest.mark.parametrize("backend,optim_name", [
    ("pyro", "Adam"),
    ("pyro", "ClippedAdam"),
    ("minipyro", "Adam"),
    ("funsor", "Adam"),
    ("funsor", "ClippedAdam"),
])
def test_optimizer(backend, optim_name, jit):

    def model(data):
        p = pyro.param("p", torch.tensor(0.5))
        pyro.sample("x", dist.Bernoulli(p), obs=data)

    def guide(data):
        pass

    data = torch.tensor(0.)
    with pyro_backend(backend):
        pyro.get_param_store().clear()
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        optimizer = getattr(optim, optim_name)({"lr": 1e-6})
        inference = infer.SVI(model, guide, optimizer, elbo)
        for i in range(2):
            inference.step(data)


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
        assert ops.allclose(actual, expected)


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
@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
def test_elbo_plate_plate(backend, outer_dim, inner_dim):
    with pyro_backend(backend):
        pyro.get_param_store().clear()
        num_particles = 1
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
            pyro.sample("w", d, infer={"enumerate": "parallel"})
            with context1:
                pyro.sample("x", d, infer={"enumerate": "parallel"})
            with context2:
                pyro.sample("y", d, infer={"enumerate": "parallel"})
            with context1, context2:
                pyro.sample("z", d, infer={"enumerate": "parallel"})

        kl_node = kl_divergence(torch.distributions.Categorical(funsor.to_data(q)),
                                torch.distributions.Categorical(funsor.to_data(p)))
        kl = (1 + outer_dim + inner_dim + outer_dim * inner_dim) * kl_node
        expected_loss = kl
        expected_grad = grad(kl, [funsor.to_data(q)])[0]

        elbo = infer.TraceEnum_ELBO(num_particles=num_particles,
                                    vectorize_particles=True,
                                    strict_enumeration_warning=True)
        elbo = elbo.differentiable_loss if backend == "pyro" else elbo
        actual_loss = funsor.to_data(elbo(model, guide))
        actual_loss.backward()
        actual_grad = funsor.to_data(pyro.param('q')).grad

        assert ops.allclose(actual_loss, expected_loss, atol=1e-5)
        assert ops.allclose(actual_grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('backend', ["pyro", "funsor"])
def test_elbo_enumerate_plates_1(backend):
    #  +-----------------+
    #  | a ----> b   M=2 |
    #  +-----------------+
    #  +-----------------+
    #  | c ----> d   N=3 |
    #  +-----------------+
    # This tests two unrelated plates.
    # Each should remain uncontracted.
    with pyro_backend(backend):
        pyro.param("probs_a",
                   torch.tensor([0.45, 0.55]),
                   constraint=constraints.simplex)
        pyro.param("probs_b",
                   torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
                   constraint=constraints.simplex)
        pyro.param("probs_c",
                   torch.tensor([0.75, 0.25]),
                   constraint=constraints.simplex)
        pyro.param("probs_d",
                   torch.tensor([[0.4, 0.6], [0.3, 0.7]]),
                   constraint=constraints.simplex)
        b_data = torch.tensor([0, 1])
        d_data = torch.tensor([0, 0, 1])

        def auto_model():
            probs_a = pyro.param("probs_a")
            probs_b = pyro.param("probs_b")
            probs_c = pyro.param("probs_c")
            probs_d = pyro.param("probs_d")
            with pyro.plate("a_axis", 2, dim=-1):
                a = pyro.sample("a", dist.Categorical(probs_a),
                                infer={"enumerate": "parallel"})
                pyro.sample("b", dist.Categorical(probs_b[a]), obs=b_data)
            with pyro.plate("c_axis", 3, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c),
                                infer={"enumerate": "parallel"})
                pyro.sample("d", dist.Categorical(probs_d[c]), obs=d_data)

        def hand_model():
            probs_a = pyro.param("probs_a")
            probs_b = pyro.param("probs_b")
            probs_c = pyro.param("probs_c")
            probs_d = pyro.param("probs_d")
            for i in range(2):
                a = pyro.sample("a_{}".format(i), dist.Categorical(probs_a),
                                infer={"enumerate": "parallel"})
                pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]), obs=b_data[i])
            for j in range(3):
                c = pyro.sample("c_{}".format(j), dist.Categorical(probs_c),
                                infer={"enumerate": "parallel"})
                pyro.sample("d_{}".format(j), dist.Categorical(probs_d[c]), obs=d_data[j])

        def guide():
            pass

        elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
        elbo = elbo.differentiable_loss if backend == "pyro" else elbo
        auto_loss = elbo(auto_model, guide)
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
        elbo = elbo.differentiable_loss if backend == "pyro" else elbo
        hand_loss = elbo(hand_model, guide)
        _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('backend', ["pyro", "funsor"])
def test_elbo_enumerate_plate_7(backend):
    #  Guide    Model
    #    a -----> b
    #    |        |
    #  +-|--------|----------------+
    #  | V        V                |
    #  | c -----> d -----> e   N=2 |
    #  +---------------------------+
    # This tests a mixture of model and guide enumeration.
    with pyro_backend(backend):
        pyro.param("model_probs_a",
                   torch.tensor([0.45, 0.55]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_b",
                   torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_c",
                   torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_d",
                   torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_e",
                   torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
                   constraint=constraints.simplex)
        pyro.param("guide_probs_a",
                   torch.tensor([0.35, 0.64]),
                   constraint=constraints.simplex)
        pyro.param("guide_probs_c",
                   torch.tensor([[0., 1.], [1., 0.]]),  # deterministic
                   constraint=constraints.simplex)

        def auto_model(data):
            probs_a = pyro.param("model_probs_a")
            probs_b = pyro.param("model_probs_b")
            probs_c = pyro.param("model_probs_c")
            probs_d = pyro.param("model_probs_d")
            probs_e = pyro.param("model_probs_e")
            a = pyro.sample("a", dist.Categorical(probs_a))
            b = pyro.sample("b", dist.Categorical(probs_b[a]),
                            infer={"enumerate": "parallel"})
            with pyro.plate("data", 2, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c[a]))
                d = pyro.sample("d", dist.Categorical(Vindex(probs_d)[b, c]),
                                infer={"enumerate": "parallel"})
                pyro.sample("obs", dist.Categorical(probs_e[d]), obs=data)

        def auto_guide(data):
            probs_a = pyro.param("guide_probs_a")
            probs_c = pyro.param("guide_probs_c")
            a = pyro.sample("a", dist.Categorical(probs_a),
                            infer={"enumerate": "parallel"})
            with pyro.plate("data", 2, dim=-1):
                pyro.sample("c", dist.Categorical(probs_c[a]))

        def hand_model(data):
            probs_a = pyro.param("model_probs_a")
            probs_b = pyro.param("model_probs_b")
            probs_c = pyro.param("model_probs_c")
            probs_d = pyro.param("model_probs_d")
            probs_e = pyro.param("model_probs_e")
            a = pyro.sample("a", dist.Categorical(probs_a))
            b = pyro.sample("b", dist.Categorical(probs_b[a]),
                            infer={"enumerate": "parallel"})
            for i in range(2):
                c = pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))
                d = pyro.sample("d_{}".format(i),
                                dist.Categorical(Vindex(probs_d)[b, c]),
                                infer={"enumerate": "parallel"})
                pyro.sample("obs_{}".format(i), dist.Categorical(probs_e[d]), obs=data[i])

        def hand_guide(data):
            probs_a = pyro.param("guide_probs_a")
            probs_c = pyro.param("guide_probs_c")
            a = pyro.sample("a", dist.Categorical(probs_a),
                            infer={"enumerate": "parallel"})
            for i in range(2):
                pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))

        data = torch.tensor([0, 0])
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
        elbo = elbo.differentiable_loss if backend == "pyro" else elbo
        auto_loss = elbo(auto_model, auto_guide, data)
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
        elbo = elbo.differentiable_loss if backend == "pyro" else elbo
        hand_loss = elbo(hand_model, hand_guide, data)
        _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.xfail(reason="missing patterns")
@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("exact", [
    True,
    xfail_param(False, reason="mixed sampling and exact not implemented yet")
], ids=["exact", "monte-carlo"])
def test_gaussian_probit_hmm_smoke(exact, jit):

    def model(data):
        T, N, D = data.shape  # time steps, individuals, features

        # Gaussian initial distribution.
        init_loc = pyro.param("init_loc", torch.zeros(D))
        init_scale = pyro.param("init_scale", 1e-2 * torch.eye(D),
                                constraint=constraints.lower_cholesky)

        # Linear dynamics with Gaussian noise.
        trans_const = pyro.param("trans_const", torch.zeros(D))
        trans_coeff = pyro.param("trans_coeff", torch.eye(D))
        noise = pyro.param("noise", 1e-2 * torch.eye(D),
                           constraint=constraints.lower_cholesky)

        obs_plate = pyro.plate("channel", D, dim=-1)
        with pyro.plate("data", N, dim=-2):
            state = None
            for t in range(T):
                # Transition.
                if t == 0:
                    loc = init_loc
                    scale_tril = init_scale
                else:
                    loc = trans_const + funsor.torch.torch_tensordot(trans_coeff, state, 1)
                    scale_tril = noise
                state = pyro.sample("state_{}".format(t),
                                    dist.MultivariateNormal(loc, scale_tril),
                                    infer={"exact": exact})

                # Factorial probit likelihood model.
                with obs_plate:
                    pyro.sample("obs_{}".format(t),
                                dist.Bernoulli(logits=state["channel"]),
                                obs=data[t])

    def guide(data):
        pass

    data = torch.distributions.Bernoulli(0.5).sample((3, 4, 2))

    with pyro_backend("funsor"):
        Elbo = infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO
        elbo = Elbo()
        adam = optim.Adam({"lr": 1e-3})
        svi = infer.SVI(model, guide, adam, elbo)
        svi.step(data)
