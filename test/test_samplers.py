from __future__ import absolute_import, division, print_function

import itertools
from collections import OrderedDict

import pytest

from funsor.domains import bint, reals
from funsor.joint import Joint
from funsor.testing import id_from_inputs, random_gaussian, random_tensor


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
def test_tensor_smoke(sample_inputs, batch_inputs, event_inputs):
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
def test_gaussian_smoke(sample_inputs, batch_inputs, event_inputs):
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
def test_joint_smoke(sample_inputs, int_event_inputs, real_event_inputs):
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
