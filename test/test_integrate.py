from collections import OrderedDict

import pytest

from funsor.domains import bint
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.montecarlo import monte_carlo
from funsor.terms import eager, lazy, moment_matching, normalize, reflect
from funsor.testing import random_tensor


@pytest.mark.parametrize('interp', [
    reflect, lazy, normalize, eager, moment_matching, monte_carlo])
def test_integrate(interp):
    log_measure = random_tensor(OrderedDict([('i', bint(2)), ('j', bint(3))]))
    integrand = random_tensor(OrderedDict([('j', bint(3)), ('k', bint(4))]))
    with interpretation(interp):
        Integrate(log_measure, integrand, frozenset(['i', 'j', 'k']))
