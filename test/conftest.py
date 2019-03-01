from __future__ import absolute_import, division, print_function

import pyro


def pytest_runtest_setup(item):
    pyro.set_rng_seed(0)
