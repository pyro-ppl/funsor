# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpyro


def pytest_runtest_setup(item):
    np.random.seed(0)
    numpyro.set_platform("cpu")
