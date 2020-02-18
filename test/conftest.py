# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from funsor.util import get_backend


def pytest_runtest_setup(item):
    backend = get_backend()

    if backend == "torch":
        import pyro

        pyro.set_rng_seed(0)
        pyro.enable_validation(True)
    elif backend == "jax":
        from jax.config import config

        config.update('jax_platform_name', 'cpu')
