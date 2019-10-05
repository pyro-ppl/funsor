from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.integrate import Integrate
from funsor.interpreter import interpretation, reinterpret
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


@dispatched_interpretation
def forget_independence(cls, *args):
    result = forget_independence.dispatch(cls, *args)
    if result is None:
        result = eager.dispatch(cls, *args)
    if result is None:
        result = normalize.dispatch(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


# TODO promote this to an eager funsor rule
@forget_independence.register(Independent, GaussianMixture, str, str, str)
def forget_independent_gaussian_mixture(fn, reals_var, bint_var, diag_var):
    tensor, gaussian = fn.terms
    info_vec = TODO
    precision = TODO
    inputs = TODO
    gaussian = Gaussian(info_vec, precision, inputs)
    return tensor + gaussian


@pytest.mark.parametrize('t_size', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('z_size,y_size', [(1, 1), (1, 2), (2, 3)])
def test_iigaussian_markov_product(t_size, z_size, y_size):
    t = Variable('t', bint(t_size))
    rhs_gaussian = (random_tensor(OrderedDict(t=bint(t_size), i=bint(y_size))) +
                    random_gaussian(OrderedDict(y=reals()))
    log_measure = Independent(Independent(rhs_gaussian, 'y', 'i'), 'y', 't')
    lhs_gaussian = (random_tensor(OrderedDict(t=bint(t_size))) +
                    random_gaussian(OrderedDict(z_prev=reals(z_size),
                                                z_curr=reals(z_size),
                                                y=reals(y_size)))
    with interpretation(lazy):
        integrand = MarkovProduct(ops.logaddeyp, ops.add, lhs_gaussian, t, {'z_prev': 'z_curr'})
    actual = Integrate(log_measure, integrand, frozenset(['y']))

    with interpretation(forget_independence):
        naive_log_measure = reinterpret(log_measure)
    assert issubclass(type(naive_log_measure), GaussianMixture)
    naive_integrand = reinterpret(integrand)
    assert issubclass(type(naive_integrand), GaussianMixture)
    expected = Integrate(naive_log_measure, naive_integrand, frozenset(['y']))
    assert_close(actual, expected)
