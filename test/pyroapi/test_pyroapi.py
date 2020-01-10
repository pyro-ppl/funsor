# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401


@pytest.yield_fixture
def backend():
    with pyro_backend("funsor"):
        yield
