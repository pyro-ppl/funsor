# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import OrderedDict
from importlib import import_module

import numpy as np
import pytest
from torch.autograd import grad

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.distribution import BACKEND_TO_DISTRIBUTIONS_BACKEND
from funsor.domains import bint, reals
from funsor.integrate import Integrate
from funsor.montecarlo import monte_carlo_interpretation
from funsor.tensor import Tensor, align_tensors
from funsor.terms import Variable
from funsor.testing import assert_close, id_from_inputs, randn, random_gaussian, random_tensor, xfail_if_not_implemented
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(get_backend() == "numpy",
                                reason="numpy does not have distributions backend")
if get_backend() != "numpy":
    dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    backend_dist = dist.dist


@pytest.mark.parametrize('sample_inputs', [
    (),
    (('s', bint(6)),),
    (('s', bint(6)), ('t', bint(7))),
], ids=id_from_inputs)
@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(4)),),
    (('b', bint(4)), ('c', bint(5))),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (('e', bint(2)),),
    (('e', bint(2)), ('f', bint(3))),
], ids=id_from_inputs)
def test_tensor_shape(sample_inputs, batch_inputs, event_inputs):
    be_inputs = OrderedDict(batch_inputs + event_inputs)
    expected_inputs = OrderedDict(sample_inputs + batch_inputs + event_inputs)
    sample_inputs = OrderedDict(sample_inputs)
    batch_inputs = OrderedDict(batch_inputs)
    event_inputs = OrderedDict(event_inputs)
    x = random_tensor(be_inputs)
    rng_key = subkey = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)

    for num_sampled in range(len(event_inputs) + 1):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            print('sampled_vars: {}'.format(', '.join(sampled_vars)))
            if rng_key is not None:
                import jax
                rng_key, subkey = jax.random.split(rng_key)

            y = x.sample(sampled_vars, sample_inputs, rng_key=subkey)
            if num_sampled == len(event_inputs):
                assert isinstance(y, (Delta, Contraction))
            if sampled_vars:
                assert dict(y.inputs) == dict(expected_inputs), sampled_vars
            else:
                assert y is x


@pytest.mark.parametrize('sample_inputs', [
    (),
    (('s', bint(3)),),
    (('s', bint(3)), ('t', bint(4))),
], ids=id_from_inputs)
@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(2)),),
    (('c', reals()),),
    (('b', bint(2)), ('c', reals())),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (('e', reals()),),
    (('e', reals()), ('f', reals(2))),
], ids=id_from_inputs)
def test_gaussian_shape(sample_inputs, batch_inputs, event_inputs):
    be_inputs = OrderedDict(batch_inputs + event_inputs)
    expected_inputs = OrderedDict(sample_inputs + batch_inputs + event_inputs)
    sample_inputs = OrderedDict(sample_inputs)
    batch_inputs = OrderedDict(batch_inputs)
    event_inputs = OrderedDict(event_inputs)
    x = random_gaussian(be_inputs)
    rng_key = subkey = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)

    xfail = False
    for num_sampled in range(len(event_inputs) + 1):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            print('sampled_vars: {}'.format(', '.join(sampled_vars)))
            try:
                if rng_key is not None:
                    import jax
                    rng_key, subkey = jax.random.split(rng_key)

                y = x.sample(sampled_vars, sample_inputs, rng_key=subkey)
            except NotImplementedError:
                xfail = True
                continue
            if num_sampled == len(event_inputs):
                assert isinstance(y, (Delta, Contraction))
            if sampled_vars:
                assert dict(y.inputs) == dict(expected_inputs), sampled_vars
            else:
                assert y is x
    if xfail:
        pytest.xfail(reason='Not implemented')


@pytest.mark.parametrize('sample_inputs', [
    (),
    (('s', bint(3)),),
    (('s', bint(3)), ('t', bint(4))),
], ids=id_from_inputs)
@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(2)),),
    (('c', reals()),),
    (('b', bint(2)), ('c', reals())),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (('e', reals()),),
    (('e', reals()), ('f', reals(2))),
], ids=id_from_inputs)
def test_transformed_gaussian_shape(sample_inputs, batch_inputs, event_inputs):
    be_inputs = OrderedDict(batch_inputs + event_inputs)
    expected_inputs = OrderedDict(sample_inputs + batch_inputs + event_inputs)
    sample_inputs = OrderedDict(sample_inputs)
    batch_inputs = OrderedDict(batch_inputs)
    event_inputs = OrderedDict(event_inputs)

    x = random_gaussian(be_inputs)
    x = x(**{name: name + '_' for name, domain in event_inputs.items()})
    x = x(**{name + '_': Variable(name, domain).log()
             for name, domain in event_inputs.items()})

    rng_key = subkey = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    xfail = False
    for num_sampled in range(len(event_inputs) + 1):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            print('sampled_vars: {}'.format(', '.join(sampled_vars)))
            try:
                if rng_key is not None:
                    import jax
                    rng_key, subkey = jax.random.split(rng_key)

                y = x.sample(sampled_vars, sample_inputs, rng_key=subkey)
            except NotImplementedError:
                xfail = True
                continue
            if num_sampled == len(event_inputs):
                assert isinstance(y, (Delta, Contraction))
            if sampled_vars:
                assert dict(y.inputs) == dict(expected_inputs), sampled_vars
            else:
                assert y is x
    if xfail:
        pytest.xfail(reason='Not implemented')


@pytest.mark.parametrize('sample_inputs', [
    (),
    (('s', bint(6)),),
    (('s', bint(6)), ('t', bint(7))),
], ids=id_from_inputs)
@pytest.mark.parametrize('int_event_inputs', [
    (),
    (('d', bint(2)),),
    (('d', bint(2)), ('e', bint(3))),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_event_inputs', [
    (('g', reals()),),
    (('g', reals()), ('h', reals(4))),
], ids=id_from_inputs)
def test_joint_shape(sample_inputs, int_event_inputs, real_event_inputs):
    event_inputs = int_event_inputs + real_event_inputs
    discrete_inputs = OrderedDict(int_event_inputs)
    gaussian_inputs = OrderedDict(event_inputs)
    expected_inputs = OrderedDict(sample_inputs + event_inputs)
    sample_inputs = OrderedDict(sample_inputs)
    event_inputs = OrderedDict(event_inputs)
    t = random_tensor(discrete_inputs)
    g = random_gaussian(gaussian_inputs)
    x = t + g  # Joint(discrete=t, gaussian=g)

    rng_key = subkey = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    xfail = False
    for num_sampled in range(len(event_inputs)):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            print('sampled_vars: {}'.format(', '.join(sampled_vars)))
            try:
                if rng_key is not None:
                    import jax
                    rng_key, subkey = jax.random.split(rng_key)

                y = x.sample(sampled_vars, sample_inputs, rng_key=subkey)
            except NotImplementedError:
                xfail = True
                continue
            if sampled_vars:
                assert dict(y.inputs) == dict(expected_inputs), sampled_vars
            else:
                assert y is x
    if xfail:
        pytest.xfail(reason='Not implemented')


@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(4)),),
    (('b', bint(2)), ('c', bint(2))),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (('e', bint(3)),),
    (('e', bint(2)), ('f', bint(2))),
], ids=id_from_inputs)
@pytest.mark.parametrize('test_grad', [False, True], ids=['value', 'grad'])
def test_tensor_distribution(event_inputs, batch_inputs, test_grad):
    num_samples = 50000
    sample_inputs = OrderedDict(n=bint(num_samples))
    be_inputs = OrderedDict(batch_inputs + event_inputs)
    batch_inputs = OrderedDict(batch_inputs)
    event_inputs = OrderedDict(event_inputs)
    sampled_vars = frozenset(event_inputs)
    p = random_tensor(be_inputs)
    p.data.requires_grad_(test_grad)

    q = p.sample(sampled_vars, sample_inputs)
    mq = p.materialize(q).reduce(ops.logaddexp, 'n')
    mq = mq.align(tuple(p.inputs))
    assert_close(mq, p, atol=0.1, rtol=None)

    if test_grad:
        _, (p_data, mq_data) = align_tensors(p, mq)
        assert p_data.shape == mq_data.shape
        probe = randn(p_data.shape)
        expected = grad((p_data.exp() * probe).sum(), [p.data])[0]
        actual = grad((mq_data.exp() * probe).sum(), [p.data])[0]
        assert_close(actual, expected, atol=0.1, rtol=None)


@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(3)),),
    (('b', bint(3)), ('c', bint(4))),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (('e', reals()),),
    (('e', reals()), ('f', reals(2))),
], ids=id_from_inputs)
def test_gaussian_distribution(event_inputs, batch_inputs):
    num_samples = 100000
    sample_inputs = OrderedDict(particle=bint(num_samples))
    be_inputs = OrderedDict(batch_inputs + event_inputs)
    batch_inputs = OrderedDict(batch_inputs)
    event_inputs = OrderedDict(event_inputs)
    sampled_vars = frozenset(event_inputs)
    p = random_gaussian(be_inputs)

    rng_key = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    q = p.sample(sampled_vars, sample_inputs, rng_key=rng_key)
    p_vars = sampled_vars
    q_vars = sampled_vars | frozenset(['particle'])
    # Check zeroth moment.
    assert_close(q.reduce(ops.logaddexp, q_vars),
                 p.reduce(ops.logaddexp, p_vars), atol=1e-6)
    for k1, d1 in event_inputs.items():
        x = Variable(k1, d1)
        # Check first moments.
        assert_close(Integrate(q, x, q_vars),
                     Integrate(p, x, p_vars), atol=0.5, rtol=0.2)
        for k2, d2 in event_inputs.items():
            y = Variable(k2, d2)
            # Check second moments.
            continue  # FIXME: Quadratic integration is not supported:
            assert_close(Integrate(q, x * y, q_vars),
                         Integrate(p, x * y, p_vars), atol=1e-2)


@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(3)),),
    (('b', bint(3)), ('c', bint(2))),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (('e', reals()), ('f', bint(3))),
    (('e', reals(2)), ('f', bint(2))),
], ids=id_from_inputs)
def test_gaussian_mixture_distribution(batch_inputs, event_inputs):
    num_samples = 100000
    sample_inputs = OrderedDict(particle=bint(num_samples))
    be_inputs = OrderedDict(batch_inputs + event_inputs)
    int_inputs = OrderedDict((k, d) for k, d in be_inputs.items()
                             if d.dtype != 'real')
    batch_inputs = OrderedDict(batch_inputs)
    event_inputs = OrderedDict(event_inputs)
    sampled_vars = frozenset(['f'])
    np.random.seed(1)  # the default random seed generate bad gaussian in jax backend
    p = random_gaussian(be_inputs) + 0.5 * random_tensor(int_inputs)
    p_marginal = p.reduce(ops.logaddexp, 'e')
    assert isinstance(p_marginal, Tensor)

    rng_key = None if get_backend() == "torch" else np.array([0, 1], dtype=np.uint32)
    q = p.sample(sampled_vars, sample_inputs, rng_key=rng_key)
    q_marginal = q.reduce(ops.logaddexp, 'e')
    q_marginal = p_marginal.materialize(q_marginal).reduce(ops.logaddexp, 'particle')
    assert isinstance(q_marginal, Tensor)
    q_marginal = q_marginal.align(tuple(p_marginal.inputs))
    assert_close(q_marginal, p_marginal, atol=0.15, rtol=None)


@pytest.mark.parametrize('moment', [0, 1, 2, 3])
def test_lognormal_distribution(moment):
    num_samples = 100000
    inputs = OrderedDict(batch=bint(10))
    loc = random_tensor(inputs)
    scale = random_tensor(inputs).exp()

    log_measure = dist.LogNormal(loc, scale)(value='x')
    probe = Variable('x', reals()) ** moment
    with monte_carlo_interpretation(particle=bint(num_samples)):
        with xfail_if_not_implemented():
            actual = Integrate(log_measure, probe, frozenset(['x']))

    samples = backend_dist.LogNormal(loc, scale).sample((num_samples,))
    expected = (samples ** moment).mean(0)
    assert_close(actual.data, expected, atol=1e-2, rtol=1e-2)
