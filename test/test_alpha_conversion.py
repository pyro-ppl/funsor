from __future__ import absolute_import, division, print_function

import itertools
from collections import OrderedDict

import numpy as np
import pytest
from six.moves import reduce

import funsor
import funsor.ops as ops
from funsor.domains import Domain, bint, reals
from funsor.interpreter import gensym, interpretation, reinterpret
from funsor.terms import Binary, Independent, Lambda, Number, Stack, Variable, reflect, sequential, to_data, to_funsor
from funsor.testing import assert_close, check_funsor, random_tensor
from funsor.torch import REDUCE_OP_TO_TORCH


def test_sample_subs():
    pass


def test_subs_reduce():
    x = random_tensor(OrderedDict([('i', bint(3)), ('j', bint(2))]), reals())
    ix = random_tensor(OrderedDict([('i', bint(3)),]), bint(2))
    ix2 = ix(i='i2')
    with interpretation(reflect):
        actual = x.reduce(ops.add, frozenset({"i"}))
    actual = actual(j=ix)
    expected = x(j=ix2).reduce(ops.add, frozenset({"i"}))(i2='i')
    assert_close(actual, expected)


@pytest.mark.parametrize('lhs_vars', [(), ('i',), ('j',), ('i', 'j')])
@pytest.mark.parametrize('rhs_vars', [(), ('i',), ('j',), ('i', 'j')])
def test_distribute_reduce(lhs_vars, rhs_vars):

    lhs_vars, rhs_vars = frozenset(lhs_vars), frozenset(rhs_vars)
    lhs = random_tensor(OrderedDict([('i', bint(3)), ('j', bint(2))]), reals())
    rhs = random_tensor(OrderedDict([('i', bint(3)), ('j', bint(2))]), reals())

    with interpretation(reflect):
        actual_lhs = lhs.reduce(ops.add, lhs_vars) if lhs_vars else lhs
        actual_rhs = rhs.reduce(ops.add, rhs_vars) if rhs_vars else rhs
    
    actual = actual_lhs * actual_rhs

    lhs_subs = {v: gensym(v) for v in lhs_vars}
    rhs_subs = {v: gensym(v) for v in rhs_vars}
    expected = (lhs(**lhs_subs) * rhs(**rhs_subs)).reduce(
        ops.add, frozenset(lhs_subs.values()) | frozenset(rhs_subs.values()))

    assert_close(actual, expected)


def test_subs_lambda():
    pass


def test_getitem_lambda():
    pass


def test_subs_independent():
    pass


def test_sample_independent():
    pass


def test_subs_gaussian():
    pass


def test_subs_contract():
    pass


def test_subs_integrate():
    pass
