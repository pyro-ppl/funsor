from __future__ import absolute_import, division, print_function

import pytest
import torch
from pyro.generic import distributions as dist
from pyro.generic import infer, optim, pyro, pyro_backend

# This file tests a variety of model,guide pairs with valid and invalid structure.
# See https://github.com/pyro-ppl/pyro/blob/0.3.1/tests/infer/test_valid_models.py


def assert_ok(backend, model, guide, *args, **kwargs):
    """
    Assert that inference works without warnings or errors.
    """
    with pyro_backend(backend):
        pyro.get_param_store().clear()
        adam = optim.Adam({"lr": 1e-6})
        elbo = infer.Trace_ELBO()
        inference = infer.SVI(model, guide, adam, elbo)
        inference.step(*args, **kwargs)


@pytest.mark.parametrize('backend', ["pyro", "minipyro", "funsor"])
def test_nonempty_model_empty_guide_ok(backend):

    def model(data):
        loc = pyro.param("loc", torch.tensor(0.0))
        pyro.sample("x", dist.Normal(loc, 1.), obs=data)

    def guide(data):
        pass

    data = torch.tensor(2.)
    assert_ok(backend, model, guide, data)
