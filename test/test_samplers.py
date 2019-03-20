from __future__ import absolute_import, division, print_function

import itertools
from collections import OrderedDict

import pytest
import torch
from torch.autograd import grad

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.joint import Joint
from funsor.terms import Variable
from funsor.testing import assert_close, id_from_inputs, random_gaussian, random_tensor
from funsor.torch import align_tensors, materialize


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

    for num_sampled in range(len(event_inputs) + 1):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            print('sampled_vars: {}'.format(', '.join(sampled_vars)))
            y = x.sample(sampled_vars, sample_inputs)
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

    xfail = False
    for num_sampled in range(len(event_inputs) + 1):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            print('sampled_vars: {}'.format(', '.join(sampled_vars)))
            try:
                y = x.sample(sampled_vars, sample_inputs)
            except NotImplementedError:
                xfail = True
                continue
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
    x = Joint(discrete=t, gaussian=g)

    xfail = False
    for num_sampled in range(len(event_inputs)):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            print('sampled_vars: {}'.format(', '.join(sampled_vars)))
            try:
                y = x.sample(sampled_vars, sample_inputs)
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

    q = p.sample(sampled_vars, sample_inputs) - ops.log(num_samples)
    mq = materialize(q).reduce(ops.logaddexp, 'n')
    mq = mq.align(tuple(p.inputs))
    assert_close(mq, p, atol=0.1, rtol=None)

    if test_grad:
        _, (p_data, mq_data) = align_tensors(p, mq)
        assert p_data.shape == mq_data.shape
        probe = torch.randn(p_data.shape)
        expected = grad((p_data.exp() * probe).sum(), [p.data])[0]
        actual = grad((mq_data.exp() * probe).sum(), [p.data])[0]
        assert_close(actual, expected, atol=0.1, rtol=None)


# This is a stub for a future PR.
def Integrate(log_measure, integrand, reduced_vars):
    pytest.xfail(reason='Integrate is not implemented')


@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(4)),),
    (('b', bint(4)), ('c', bint(5))),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (('e', reals()),),
    (('e', reals()), ('f', reals(2))),
], ids=id_from_inputs)
def test_gaussian_distribution(event_inputs, batch_inputs):
    num_samples = 10000
    sample_inputs = OrderedDict(n=bint(num_samples))
    be_inputs = OrderedDict(batch_inputs + event_inputs)
    batch_inputs = OrderedDict(batch_inputs)
    event_inputs = OrderedDict(event_inputs)
    sampled_vars = frozenset(event_inputs)
    p = random_gaussian(be_inputs)

    q = p.sample(sampled_vars, sample_inputs) - ops.log(num_samples)
    p_vars = sampled_vars
    q_vars = sampled_vars | frozenset(['n'])
    # Check zeroth moment.
    assert_close(q.reduce(ops.logaddexp, q_vars),
                 p.reduce(ops.logaddexp, p_vars), atol=1e-6, rtol=None)
    for k1, d1 in event_inputs.items():
        x = Variable(k1, d1)
        # Check first moments.
        assert_close(Integrate(q, x, q_vars),
                     Integrate(p, x, p_vars), atol=1e-2, rtol=None)
        for k2, d2 in event_inputs.item():
            y = Variable(k2, d2)
            # Check second moments.
            assert_close(Integrate(q, x * y, q_vars),
                         Integrate(p, x * y, p_vars), atol=1e-2, rtol=None)
