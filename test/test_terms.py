from __future__ import absolute_import, division, print_function

import itertools

import numpy as np
import pytest
from six.moves import reduce

import funsor
import funsor.ops as ops
from funsor.domains import Domain, bint, reals
from funsor.interpreter import interpretation
from funsor.terms import Binary, Lambda, Number, Stack, Variable, sequential, to_data, to_funsor
from funsor.testing import check_funsor
from funsor.torch import REDUCE_OP_TO_TORCH

np.seterr(all='ignore')


def test_to_funsor():
    assert to_funsor(0) is Number(0)


@pytest.mark.parametrize('x', ["foo", list(), tuple(), set(), dict()])
def test_to_funsor_error(x):
    with pytest.raises(ValueError):
        to_funsor(x)


def test_to_data():
    actual = to_data(Number(0.))
    expected = 0.
    assert type(actual) == type(expected)
    assert actual == expected


def test_to_data_error():
    with pytest.raises(ValueError):
        to_data(Variable('x', reals()))
    with pytest.raises(ValueError):
        to_data(Variable('y', bint(12)))


def test_cons_hash():
    assert Variable('x', bint(3)) is Variable('x', bint(3))
    assert Variable('x', reals()) is Variable('x', reals())
    assert Variable('x', reals()) is not Variable('x', bint(3))
    assert Number(0, 3) is Number(0, 3)
    assert Number(0.) is Number(0.)
    assert Number(0.) is not Number(0, 3)


@pytest.mark.parametrize('expr', [
    "Variable('x', bint(3))",
    "Variable('x', reals())",
    "Number(0.)",
    "Number(1, dtype=10)",
    "-Variable('x', reals())",
    "Variable('x', reals()) + Variable('y', reals())",
    "Variable('x', reals())(x=Number(0.))",
])
def test_reinterpret(expr):
    x = eval(expr)
    assert funsor.reinterpret(x) is x


@pytest.mark.parametrize('domain', [bint(3), reals()])
def test_variable(domain):
    x = Variable('x', domain)
    check_funsor(x, {'x': domain}, domain)
    assert Variable('x', domain) is x
    assert x('x') is x
    y = Variable('y', domain)
    assert x('y') is y
    assert x(x='y') is y
    assert x(x=y) is y
    x4 = Variable('x', bint(4))
    assert x4 is not x
    assert x4('x') is x4
    assert x(y=x4) is x

    xp1 = x + 1.
    assert xp1(x=2.) == 3.


def test_substitute():
    x = Variable('x', reals())
    y = Variable('y', reals())
    z = Variable('z', reals())

    f = x * y + x * z
    assert isinstance(f, Binary)
    assert f.op is ops.add

    assert f(y=2) is x * 2 + x * z
    assert f(z=2) is x * y + x * 2
    assert f(y=x) is x * x + x * z
    assert f(x=y) is y * y + y * z
    assert f(y=z, z=y) is x * z + x * y
    assert f(x=y, y=z, z=x) is y * z + y * x


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(ops, symbol)(x)


@pytest.mark.parametrize('data', [0, 0.5, 1])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p',
])
def test_unary(symbol, data):
    dtype = 'real'
    if symbol == '~':
        data = bool(data)
        dtype = 2
    expected_data = unary_eval(symbol, data)

    x = Number(data, dtype)
    actual = unary_eval(symbol, x)
    check_funsor(actual, {}, Domain((), dtype), expected_data)


BINARY_OPS = [
    '+', '-', '*', '/', '**', '==', '!=', '<', '<=', '>', '>=',
    'min', 'max',
]
BOOLEAN_OPS = ['&', '|', '^']


def binary_eval(symbol, x, y):
    if symbol == 'min':
        return ops.min(x, y)
    if symbol == 'max':
        return ops.max(x, y)
    return eval('x {} y'.format(symbol))


@pytest.mark.parametrize('data1', [0, 0.2, 1])
@pytest.mark.parametrize('data2', [0, 0.8, 1])
@pytest.mark.parametrize('symbol', BINARY_OPS + BOOLEAN_OPS)
def test_binary(symbol, data1, data2):
    dtype = 'real'
    if symbol in BOOLEAN_OPS:
        dtype = 2
        data1 = bool(data1)
        data2 = bool(data2)
    try:
        expected_data = binary_eval(symbol, data1, data2)
    except ZeroDivisionError:
        return

    x1 = Number(data1, dtype)
    x2 = Number(data2, dtype)
    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, {}, Domain((), dtype), expected_data)


@pytest.mark.parametrize('op', REDUCE_OP_TO_TORCH,
                         ids=[op.__name__ for op in REDUCE_OP_TO_TORCH])
def test_reduce_all(op):
    x = Variable('x', bint(2))
    y = Variable('y', bint(3))
    z = Variable('z', bint(4))
    f = x * y + z
    dtype = f.dtype
    check_funsor(f, {'x': bint(2), 'y': bint(3), 'z': bint(4)}, Domain((), dtype))
    if op is ops.logaddexp:
        pytest.skip()

    with interpretation(sequential):
        actual = f.reduce(op)

    values = [f(x=i, y=j, z=k)
              for i in x.output
              for j in y.output
              for k in z.output]
    expected = reduce(op, values)
    assert actual == expected


@pytest.mark.parametrize('reduced_vars', [
    reduced_vars
    for num_reduced in range(3 + 1)
    for reduced_vars in itertools.combinations('xyz', num_reduced)
])
@pytest.mark.parametrize('op', REDUCE_OP_TO_TORCH,
                         ids=[op.__name__ for op in REDUCE_OP_TO_TORCH])
def test_reduce_subset(op, reduced_vars):
    reduced_vars = frozenset(reduced_vars)
    x = Variable('x', bint(2))
    y = Variable('y', bint(3))
    z = Variable('z', bint(4))
    f = x * y + z
    dtype = f.dtype
    check_funsor(f, {'x': bint(2), 'y': bint(3), 'z': bint(4)}, Domain((), dtype))
    if op is ops.logaddexp:
        pytest.skip()

    with interpretation(sequential):
        actual = f.reduce(op, reduced_vars)

    expected = f
    for v in [x, y, z]:
        if v.name in reduced_vars:
            expected = reduce(op, [expected(**{v.name: i}) for i in v.output])

    check_funsor(actual, expected.inputs, expected.output)
    # TODO check data
    if not reduced_vars:
        assert actual is f


@pytest.mark.parametrize('base_shape', [(), (4,), (3, 2)], ids=str)
def test_lambda(base_shape):
    z = Variable('z', reals(*base_shape))
    i = Variable('i', bint(5))
    j = Variable('j', bint(7))

    zi = Lambda(i, z)
    assert zi.output.shape == (5,) + base_shape
    assert zi[i] is z

    zj = Lambda(j, z)
    assert zj.output.shape == (7,) + base_shape
    assert zj[j] is z

    zij = Lambda(j, zi)
    assert zij.output.shape == (7, 5) + base_shape
    assert zij[j] is zi
    assert zij[:, i] is zj
    assert zij[j, i] is z


def test_stack_simple():
    x = Number(0.)
    y = Number(1.)
    z = Number(4.)

    xyz = Stack((x, y, z), 'i')
    check_funsor(xyz, {'i': bint(3)}, reals())

    assert xyz(i=Number(0, 3)) is x
    assert xyz(i=Number(1, 3)) is y
    assert xyz(i=Number(2, 3)) is z
    assert xyz.reduce(ops.add, 'i') == 5.


def test_stack_subs():
    x = Variable('x', reals())
    y = Variable('y', reals())
    z = Variable('z', reals())
    j = Variable('j', bint(3))

    f = Stack((Number(0), x, y * z), 'i')
    check_funsor(f, {'i': bint(3), 'x': reals(), 'y': reals(), 'z': reals()},
                 reals())

    assert f(i=Number(0, 3)) is Number(0)
    assert f(i=Number(1, 3)) is x
    assert f(i=Number(2, 3)) is y * z
    assert f(i=j) is Stack((Number(0), x, y * z), 'j')
    assert f(i='j') is Stack((Number(0), x, y * z), 'j')
    assert f.reduce(ops.add, 'i') is Number(0) + x + (y * z)

    assert f(x=0) is Stack((Number(0), Number(0), y * z), 'i')
    assert f(y=x) is Stack((Number(0), x, x * z), 'i')
    assert f(x=0, y=x) is Stack((Number(0), Number(0), x * z), 'i')
    assert f(x=0, y=x, i=Number(2, 3)) is x * z
    assert f(x=0, i=j) is Stack((Number(0), Number(0), y * z), 'j')
    assert f(x=0, i='j') is Stack((Number(0), Number(0), y * z), 'j')


def test_align_simple():
    x = Variable('x', reals())
    y = Variable('y', reals())
    z = Variable('z', reals())
    f = z + y * x
    assert tuple(f.inputs) == ('z', 'y', 'x')
    g = f.align(('x', 'y', 'z'))
    assert tuple(g.inputs) == ('x', 'y', 'z')
    for k, v in f.inputs.items():
        assert g.inputs[k] == v
    assert f(x=1, y=2, z=3) == g(x=1, y=2, z=3)
