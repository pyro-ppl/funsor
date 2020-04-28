# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import funsor.util
funsor.util.set_backend("torch")


def _disallow_set_backend(*args):
    raise ValueError("set_backend() cannot be called during tests")


def pytest_runtest_setup(item):
    np.random.seed(0)
    backend = funsor.util.get_backend()
    if backend == "torch":
        import pyro

        pyro.set_rng_seed(0)
        pyro.enable_validation(True)
    elif backend == "jax":
        from jax.config import config

        config.update('jax_platform_name', 'cpu')

    funsor.util.set_backend = _disallow_set_backend
