# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
from jax.config import config

config.update('jax_platform_name', 'cpu')


def pytest_runtest_setup(item):
    pyro.set_rng_seed(0)
    pyro.enable_validation(True)
