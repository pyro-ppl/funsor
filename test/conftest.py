import pyro


def pytest_runtest_setup(item):
    pyro.set_rng_seed(0)
    pyro.enable_validation(True)
