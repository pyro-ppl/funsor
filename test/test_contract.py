from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest

from funsor.contract import _partition
from funsor.domains import bint
from funsor.testing import random_tensor


@pytest.mark.parametrize('inputs,dims,expected_num_components', [
    ([''], set(), 1),
    (['a'], set(), 1),
    (['a'], set('a'), 1),
    (['a', 'a'], set(), 2),
    (['a', 'a'], set('a'), 1),
    (['a', 'a', 'b', 'b'], set(), 4),
    (['a', 'a', 'b', 'b'], set('a'), 3),
    (['a', 'a', 'b', 'b'], set('b'), 3),
    (['a', 'a', 'b', 'b'], set('ab'), 2),
    (['a', 'ab', 'b'], set(), 3),
    (['a', 'ab', 'b'], set('a'), 2),
    (['a', 'ab', 'b'], set('b'), 2),
    (['a', 'ab', 'b'], set('ab'), 1),
    (['a', 'ab', 'bc', 'c'], set(), 4),
    (['a', 'ab', 'bc', 'c'], set('c'), 3),
    (['a', 'ab', 'bc', 'c'], set('b'), 3),
    (['a', 'ab', 'bc', 'c'], set('a'), 3),
    (['a', 'ab', 'bc', 'c'], set('ac'), 2),
    (['a', 'ab', 'bc', 'c'], set('abc'), 1),
])
def test_partition(inputs, dims, expected_num_components):
    sizes = dict(zip('abc', [2, 3, 4]))
    terms = [random_tensor(OrderedDict((s, bint(sizes[s])) for s in input_))
             for input_ in inputs]
    components = list(_partition(terms, dims))

    # Check that result is a partition.
    expected_terms = sorted(terms, key=id)
    actual_terms = sorted((x for c in components for x in c[0]), key=id)
    assert actual_terms == expected_terms
    assert dims == set.union(set(), *(c[1] for c in components))

    # Check that the partition is not too coarse.
    assert len(components) == expected_num_components

    # Check that partition is not too fine.
    component_dict = {x: i for i, (terms, _) in enumerate(components) for x in terms}
    for x in terms:
        for y in terms:
            if x is not y:
                if dims.intersection(x.inputs, y.inputs):
                    assert component_dict[x] == component_dict[y]
