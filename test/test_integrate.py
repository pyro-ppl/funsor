# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpy as np
import pytest

from funsor import ops
from funsor.domains import Bint
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.montecarlo import MonteCarlo
from funsor.terms import Variable, eager, lazy, moment_matching, normalize, reflect
from funsor.testing import assert_close, random_tensor


@pytest.mark.parametrize('interp', [
    reflect, lazy, normalize, eager, moment_matching,
    MonteCarlo(rng_key=np.array([0, 0], dtype=np.uint32)),
])
def test_integrate(interp):
    log_measure = random_tensor(OrderedDict([('i', Bint[2]), ('j', Bint[3])]))
    integrand = random_tensor(OrderedDict([('j', Bint[3]), ('k', Bint[4])]))
    with interpretation(interp):
        Integrate(log_measure, integrand, {'i', 'j', 'k'})


def test_syntactic_sugar():
    i = Variable("i", Bint[3])
    log_measure = random_tensor(OrderedDict(i=Bint[3]))
    integrand = random_tensor(OrderedDict(i=Bint[3]))
    expected = (log_measure.exp() * integrand).reduce(ops.add, "i")
    assert_close(Integrate(log_measure, integrand, "i"), expected)
    assert_close(Integrate(log_measure, integrand, {"i"}), expected)
    assert_close(Integrate(log_measure, integrand, frozenset(["i"])), expected)
    assert_close(Integrate(log_measure, integrand, i), expected)
    assert_close(Integrate(log_measure, integrand, {i}), expected)
    assert_close(Integrate(log_measure, integrand, frozenset([i])), expected)
