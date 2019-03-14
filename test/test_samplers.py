from __future__ import absolute_import, division, print_function

import itertools
from collections import OrderedDict

import pytest

from funsor.domains import bint
from funsor.testing import id_from_inputs, random_tensor


@pytest.mark.parametrize('sample_inputs', [
    (),
    (('s', bint(2)),),
    (('s', bint(2)), ('t', bint(3))),
], ids=id_from_inputs)
@pytest.mark.parametrize('batch_inputs', [
    (),
    (('b', bint(2)),),
    (('b', bint(2)), ('c', bint(3))),
], ids=id_from_inputs)
@pytest.mark.parametrize('event_inputs', [
    (),
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
    for num_sampled in range(len(event_inputs)):
        for sampled_vars in itertools.combinations(list(event_inputs), num_sampled):
            sampled_vars = frozenset(sampled_vars)
            y = x.sample(sampled_vars, sample_inputs)
            if sampled_vars:
                assert dict(y.inputs) == dict(expected_inputs), sampled_vars
            else:
                assert y is x
