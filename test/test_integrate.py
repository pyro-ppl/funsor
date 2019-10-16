from collections import OrderedDict

import pytest

from funsor import ops
from funsor.domains import bint
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.montecarlo import monte_carlo
from funsor.terms import Variable, eager, lazy, moment_matching, normalize, reflect
from funsor.testing import assert_close, random_tensor


@pytest.mark.parametrize('interp', [
    reflect, lazy, normalize, eager, moment_matching, monte_carlo])
def test_integrate(interp):
    log_measure = random_tensor(OrderedDict([('i', bint(2)), ('j', bint(3))]))
    integrand = random_tensor(OrderedDict([('j', bint(3)), ('k', bint(4))]))
    with interpretation(interp):
        Integrate(log_measure, integrand, {'i', 'j', 'k'})


def test_syntactic_sugar():
    i = Variable("i", bint(3))
    log_measure = random_tensor(OrderedDict(i=bint(3)))
    integrand = random_tensor(OrderedDict(i=bint(3)))
    expected = (log_measure.exp() * integrand).reduce(ops.add, "i")
    assert_close(Integrate(log_measure, integrand, "i"), expected)
    assert_close(Integrate(log_measure, integrand, {"i"}), expected)
    assert_close(Integrate(log_measure, integrand, frozenset(["i"])), expected)
    assert_close(Integrate(log_measure, integrand, i), expected)
    assert_close(Integrate(log_measure, integrand, {i}), expected)
    assert_close(Integrate(log_measure, integrand, frozenset([i])), expected)
