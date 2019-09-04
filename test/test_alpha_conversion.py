from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.interpreter import gensym, interpretation, reinterpret
from funsor.terms import Independent, Lambda, Variable, reflect
from funsor.testing import assert_close, check_funsor, random_tensor


def test_sample_subs_smoke():
    x = random_tensor(OrderedDict([('i', bint(3)), ('j', bint(2))]), reals())
    with interpretation(reflect):
        z = x(i=1)
    actual = z.sample(frozenset({"j"}), OrderedDict({"i": bint(4)}))
    check_funsor(actual, {"j": bint(2), "i": bint(4)}, reals())


def test_subs_reduce():
    x = random_tensor(OrderedDict([('i', bint(3)), ('j', bint(2))]), reals())
    ix = random_tensor(OrderedDict([('i', bint(3))]), bint(2))
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

    actual = reinterpret(actual_lhs * actual_rhs)

    lhs_subs = {v: gensym(v) for v in lhs_vars}
    rhs_subs = {v: gensym(v) for v in rhs_vars}
    expected = (lhs(**lhs_subs) * rhs(**rhs_subs)).reduce(
        ops.add, frozenset(lhs_subs.values()) | frozenset(rhs_subs.values()))

    assert_close(actual, expected)


def test_subs_lambda():
    z = Variable('z', reals())
    i = Variable('i', bint(5))
    ix = random_tensor(OrderedDict([('i', bint(5))]), reals())
    actual = Lambda(i, z)(z=ix)
    expected = Lambda(i(i='j'), z(z=ix))
    check_funsor(actual, expected.inputs, expected.output)
    assert_close(actual, expected)


def test_slice_lambda():
    z = Variable('z', reals())
    i = Variable('i', bint(5))
    j = Variable('j', bint(7))
    zi = Lambda(i, z)
    zj = Lambda(j, z)
    zij = Lambda(j, zi)
    zj2 = zij[:, i]
    check_funsor(zj2, zj.inputs, zj.output)


def test_subs_independent():
    f = Variable('x', reals(4, 5)) + random_tensor(OrderedDict(i=bint(3)))

    actual = Independent(f, 'x', 'i')
    assert 'i' not in actual.inputs

    y = Variable('y', reals(3, 4, 5))
    fsub = y + (0. * random_tensor(OrderedDict(i=bint(7))))
    actual = actual(x=fsub)
    assert actual.inputs['i'] == bint(7)

    expected = f(x=y['i']).reduce(ops.add, 'i')

    data = random_tensor(OrderedDict(i=bint(7)), y.output)
    assert_close(actual(y=data), expected(y=data))


@pytest.mark.xfail(reason="Independent not quite compatible with sample")
def test_sample_independent():
    f = Variable('x', reals(4, 5)) + random_tensor(OrderedDict(i=bint(3)))
    actual = Independent(f, 'x', 'i')
    assert actual.sample('i')
    assert actual.sample('j', {'i': 2})
