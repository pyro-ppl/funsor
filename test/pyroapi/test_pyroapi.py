import pytest
from pyroapi import pyro_backend
from pyroapi.tests import *  # noqa F401


@pytest.fixture(params=["funsor"])
def backend(request):
    with pyro_backend(request.param):
        yield
