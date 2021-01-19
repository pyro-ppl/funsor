# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
import io
import itertools
import pickle
import typing
from collections import OrderedDict
from functools import reduce

import numpy as np
import pytest

import funsor
import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.domains import Array, Bint, Product, Real, Reals
from funsor.interpreter import interpretation, reinterpret
from funsor.tensor import REDUCE_OP_TO_NUMERIC
from funsor.terms import (
    Binary,
    Cat,
    Funsor,
    Independent,
    Lambda,
    Number,
    Reduce,
    Slice,
    Stack,
    Subs,
    Tuple,
    Variable,
    eager,
    eager_or_die,
    lazy,
    normalize,
    reflect,
    sequential,
    to_data,
    to_funsor
)
from funsor.testing import assert_close, check_funsor, random_tensor

assert Binary  # flake8
assert Subs  # flake8
assert Contraction  # flake8
assert Reals  # flake8

np.seterr(all='ignore')


# have to make this deterministic for pytest collection to work
SORTED_REDUCE_OP_TO_NUMERIC = OrderedDict(
    sorted(REDUCE_OP_TO_NUMERIC.items(), key=lambda ops_: ops_[0].__name__))


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
        to_data(Variable('x', Real))
    with pytest.raises(ValueError):
        to_data(Variable('y', Bint[12]))


def test_cons_hash():
    assert Variable('x', Bint[3]) is Variable('x', Bint[3])
    assert Variable('x', Real) is Variable('x', Real)
    assert Variable('x', Real) is not Variable('x', Bint[3])
    assert Number(0, 3) is Number(0, 3)
    assert Number(0.) is Number(0.)
    assert Number(0.) is not Number(0, 3)
    assert Slice('x', 10) is Slice('x', 10)
    assert Slice('x', 10) is Slice('x', 0, 10)
    assert Slice('x', 10, 10) is not Slice('x', 0, 10)
    assert Slice('x', 2, 10, 1) is Slice('x', 2, 10)


def check_quote(x):
    s = funsor.quote(x)
    assert isinstance(s, str)
    y = eval(s)
    assert x is y


@pytest.mark.parametrize('interp', [reflect, lazy, normalize, eager], ids=lambda i: i.__name__)
def test_quote(interp):
    with interpretation(interp):
        x = Variable('x', Bint[8])
        check_quote(x)

        y = Variable('y', Reals[8, 3, 3])
        check_quote(y)
        check_quote(y[x])

        z = Stack('i', (Number(0), Variable('z', Real)))
        check_quote(z)
        check_quote(z(i=0))
        check_quote(z(i=Slice('i', 0, 1, 1, 2)))
        check_quote(z.reduce(ops.add, 'i'))
        check_quote(Cat('i', (z, z, z)))
        check_quote(Lambda(Variable('i', Bint[2]), z))


EXPR_STRINGS = [
    "Variable('x', Bint[3])",
    "Variable('x', Real)",
    "Number(0.)",
    "Number(1, dtype=10)",
    "-Variable('x', Real)",
    "Variable('x', Reals[3])[Variable('i', Bint[3])]",
    "Variable('x', Reals[2, 2]).reshape((4,))",
    "Variable('x', Real) + Variable('y', Real)",
    "Variable('x', Real)(x=Number(0.))",
    "Number(1) / Variable('x', Real)",
    "Stack('i', (Number(0), Variable('z', Real)))",
    "Cat('i', (Stack('i', (Number(0),)), Stack('i', (Number(1), Number(2)))))",
    "Stack('t', (Number(1), Variable('x', Real))).reduce(ops.logaddexp, 't')",
]


@pytest.mark.parametrize('expr', EXPR_STRINGS)
def test_copy_immutable(expr):
    x = eval(expr)
    assert copy.copy(x) is x


@pytest.mark.parametrize('expr', EXPR_STRINGS)
def test_deepcopy_immutable(expr):
    x = eval(expr)
    assert copy.deepcopy(x) is x


@pytest.mark.parametrize('expr', EXPR_STRINGS)
def test_pickle(expr):
    x = eval(expr)
    f = io.BytesIO()
    pickle.dump(x, f)
    f.seek(0)
    y = pickle.load(f)
    assert y is x


@pytest.mark.parametrize('expr', EXPR_STRINGS)
def test_reinterpret(expr):
    x = eval(expr)
    assert funsor.reinterpret(x) is x


@pytest.mark.parametrize('expr', EXPR_STRINGS)
def test_type_hints(expr):
    x = eval(expr)
    inputs = typing.get_type_hints(x)
    output = inputs.pop("return")
    assert inputs == dict(x.inputs)
    assert output == x.output


@pytest.mark.parametrize("expr", [
    "Variable('x', Real)",
    "Number(1)",
    "Number(1).log()",
    "Number(1) + Number(2)",
    "Stack('t', (Number(1), Number(2)))",
    "Stack('t', (Number(1), Number(2))).reduce(ops.add, 't')",
])
def test_eager_or_die_ok(expr):
    with interpretation(eager_or_die):
        eval(expr)


@pytest.mark.parametrize("expr", [
    "Variable('x', Real).log()",
    "Number(1) / Variable('x', Real)",
    "Variable('x', Real) ** Number(2)",
    "Stack('t', (Number(1), Variable('x', Real))).reduce(ops.logaddexp, 't')",
])
def test_eager_or_die_error(expr):
    with interpretation(eager_or_die):
        with pytest.raises(NotImplementedError):
            eval(expr)


@pytest.mark.parametrize('domain', [Bint[3], Real])
def test_variable(domain):
    x = Variable('x', domain)
    check_funsor(x, {'x': domain}, domain)
    assert Variable('x', domain) is x
    assert x('x') is x
    y = Variable('y', domain)
    assert x('y') is y
    assert x(x='y') is y
    assert x(x=y) is y
    x4 = Variable('x', Bint[4])
    assert x4 is not x
    assert x4('x') is x4
    assert x(y=x4) is x

    xp1 = x + 1.
    assert xp1(x=2.) == 3.


def test_substitute():
    x = Variable('x', Real)
    y = Variable('y', Real)
    z = Variable('z', Real)

    f = x * y + x * z

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
    '~', '-', 'atanh', 'abs', 'sqrt', 'exp', 'log', 'log1p', 'sigmoid', 'tanh',
])
def test_unary(symbol, data):
    dtype = 'real'
    if symbol == '~':
        data = bool(data)
        dtype = 2
    if symbol == 'atanh':
        data = min(data, 0.99)
    expected_data = unary_eval(symbol, data)

    x = Number(data, dtype)
    actual = unary_eval(symbol, x)
    check_funsor(actual, {}, Array[dtype, ()], expected_data)


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
    check_funsor(actual, {}, Array[dtype, ()], expected_data)
    with interpretation(normalize):
        actual_reflect = binary_eval(symbol, x1, x2)
    assert actual.output == actual_reflect.output


@pytest.mark.parametrize('op', SORTED_REDUCE_OP_TO_NUMERIC,
                         ids=[op.__name__ for op in SORTED_REDUCE_OP_TO_NUMERIC])
def test_reduce_all(op):
    x = Variable('x', Bint[2])
    y = Variable('y', Bint[3])
    z = Variable('z', Bint[4])
    if isinstance(op, ops.LogaddexpOp):
        pytest.skip()  # not defined for integers

    with interpretation(sequential):
        f = x * y + z
        dtype = f.dtype
        check_funsor(f, {'x': Bint[2], 'y': Bint[3], 'z': Bint[4]}, Array[dtype, ()])
        actual = f.reduce(op)

    with interpretation(sequential):
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
@pytest.mark.parametrize('op', SORTED_REDUCE_OP_TO_NUMERIC,
                         ids=[op.__name__ for op in SORTED_REDUCE_OP_TO_NUMERIC])
def test_reduce_subset(op, reduced_vars):
    reduced_vars = frozenset(reduced_vars)
    x = Variable('x', Bint[2])
    y = Variable('y', Bint[3])
    z = Variable('z', Bint[4])
    f = x * y + z
    dtype = f.dtype
    check_funsor(f, {'x': Bint[2], 'y': Bint[3], 'z': Bint[4]}, Array[dtype, ()])
    if isinstance(op, ops.LogaddexpOp):
        pytest.skip()  # not defined for integers

    with interpretation(sequential):
        actual = f.reduce(op, reduced_vars)
        expected = f
        for v in [x, y, z]:
            if v.name in reduced_vars:
                expected = reduce(op, [expected(**{v.name: i}) for i in v.output])

    try:
        check_funsor(actual, expected.inputs, expected.output)
    except AssertionError:
        assert type(actual).__origin__ == type(expected).__origin__
        assert actual.inputs == expected.inputs
        assert actual.output.dtype != 'real' and expected.output.dtype != 'real'
        pytest.xfail(reason="bound inference not quite right")

    # TODO check data
    if not reduced_vars:
        assert actual is f


def test_reduce_syntactic_sugar():
    i = Variable("i", Bint[3])
    x = Stack("i", (Number(1), Number(2), Number(3)))
    expected = Number(1 + 2 + 3)
    assert x.reduce(ops.add) is expected
    assert x.reduce(ops.add, "i") is expected
    assert x.reduce(ops.add, {"i"}) is expected
    assert x.reduce(ops.add, frozenset(["i"])) is expected
    assert x.reduce(ops.add, i) is expected
    assert x.reduce(ops.add, {i}) is expected
    assert x.reduce(ops.add, frozenset([i])) is expected


def test_reduce_constant():
    x = Number(1)
    i = Variable("i", Bint[4])
    assert x.reduce(ops.add, i) == Number(4)


def test_reduce_variable():
    x = Variable("x", Real)
    i = Variable("i", Bint[4])
    assert x.reduce(ops.add, i) is x * 4


def test_slice():
    t_slice = Slice("t", 10)

    s_slice = t_slice(t="s")
    assert isinstance(s_slice, Slice)
    assert s_slice.slice == t_slice.slice
    assert s_slice(s="t") is t_slice

    assert t_slice(t=0) is Number(0, 10)
    assert t_slice(t=1) is Number(1, 10)
    assert t_slice(t=2) is Number(2, 10)
    assert t_slice(t=t_slice) is t_slice


@pytest.mark.parametrize('base_shape', [(), (4,), (3, 2)], ids=str)
def test_lambda(base_shape):
    z = Variable('z', Reals[base_shape])
    i = Variable('i', Bint[5])
    j = Variable('j', Bint[7])

    zi = Lambda(i, z)
    assert zi.output.shape == (5,) + base_shape
    assert zi[i] is z

    zj = Lambda(j, z)
    assert zj.output.shape == (7,) + base_shape
    assert zj[j] is z

    zij = Lambda(j, zi)
    assert zij.output.shape == (7, 5) + base_shape
    assert zij[j] is zi
    assert zij[j, i] is z
    # assert zij[:, i] is zj  # XXX this was disabled by alpha-renaming
    check_funsor(zij[:, i], zj.inputs, zj.output)


@pytest.mark.parametrize("dtype", ["real", 2, 3])
def test_independent(dtype):
    f = Variable('x_i', Array[dtype, (4, 5)]) + random_tensor(OrderedDict(i=Bint[3]), output=Array[dtype, ()])
    assert f.inputs['x_i'] == Array[dtype, (4, 5)]
    assert f.inputs['i'] == Bint[3]

    actual = Independent(f, 'x', 'i', 'x_i')
    assert actual.inputs['x'] == Array[dtype, (3, 4, 5)]
    assert 'i' not in actual.inputs

    x = Variable('x', Array[dtype, (3, 4, 5)])
    expected = f(x_i=x['i']).reduce(ops.add, 'i')
    assert actual.inputs == expected.inputs
    assert actual.output == expected.output

    data = random_tensor(OrderedDict(), x.output)
    assert_close(actual(data), expected(data), atol=1e-5, rtol=1e-5)

    renamed = actual(x='y')
    assert isinstance(renamed, Independent)
    assert_close(renamed(y=data), expected(x=data), atol=1e-5, rtol=1e-5)

    # Ensure it's ok for .reals_var and .diag_var to be the same.
    renamed = actual(x='x_i')
    assert isinstance(renamed, Independent)
    assert_close(renamed(x_i=data), expected(x=data), atol=1e-5, rtol=1e-5)


def test_stack_simple():
    x = Number(0.)
    y = Number(1.)
    z = Number(4.)

    xyz = Stack('i', (x, y, z))
    check_funsor(xyz, {'i': Bint[3]}, Real)

    assert xyz(i=Number(0, 3)) is x
    assert xyz(i=Number(1, 3)) is y
    assert xyz(i=Number(2, 3)) is z
    assert xyz.reduce(ops.add, 'i') == 5.


def test_stack_subs():
    x = Variable('x', Real)
    y = Variable('y', Real)
    z = Variable('z', Real)
    j = Variable('j', Bint[3])

    f = Stack('i', (Number(0), x, y * z))
    check_funsor(f, {'i': Bint[3], 'x': Real, 'y': Real, 'z': Real},
                 Real)

    assert f(i=Number(0, 3)) is Number(0)
    assert f(i=Number(1, 3)) is x
    assert f(i=Number(2, 3)) is y * z
    assert f(i=j) is Stack('j', (Number(0), x, y * z))
    assert f(i='j') is Stack('j', (Number(0), x, y * z))
    assert f.reduce(ops.add, 'i') is Number(0) + x + (y * z)

    assert f(x=0) is Stack('i', (Number(0), Number(0), y * z))
    assert f(y=x) is Stack('i', (Number(0), x, x * z))
    assert f(x=0, y=x) is Stack('i', (Number(0), Number(0), x * z))
    assert f(x=0, y=x, i=Number(2, 3)) is x * z
    assert f(x=0, i=j) is Stack('j', (Number(0), Number(0), y * z))
    assert f(x=0, i='j') is Stack('j', (Number(0), Number(0), y * z))


@pytest.mark.parametrize("start,stop", [(0, 1), (0, 2), (0, 10), (1, 2), (1, 10), (2, 10)])
@pytest.mark.parametrize("step", [1, 2, 5, 10])
def test_stack_slice(start, stop, step):
    xs = tuple(map(Number, range(10)))
    actual = Stack('i', xs)(i=Slice('j', start, stop, step, dtype=10))
    expected = Stack('j', xs[start: stop: step])
    assert type(actual) == type(expected)
    assert actual.name == expected.name
    assert actual.parts == expected.parts


def test_cat_simple():
    x = Stack('i', (Number(0), Number(1), Number(2)))
    y = Stack('i', (Number(3), Number(4)))

    assert Cat('i', (x,)) is x
    assert Cat('i', (y,)) is y

    xy = Cat('i', (x, y))
    assert xy.inputs == OrderedDict(i=Bint[5])
    assert xy.name == 'i'
    for i in range(5):
        assert xy(i=i) is Number(i)


def test_align_simple():
    x = Variable('x', Real)
    y = Variable('y', Real)
    z = Variable('z', Real)
    f = z + y * x
    assert tuple(f.inputs) == ('z', 'y', 'x')
    g = f.align(('x', 'y', 'z'))
    assert tuple(g.inputs) == ('x', 'y', 'z')
    for k, v in f.inputs.items():
        assert g.inputs[k] == v
    assert f(x=1, y=2, z=3) == g(x=1, y=2, z=3)


@pytest.mark.parametrize("subcls_expr,cls_expr", [
    ("Reduce", "Reduce"),
    ("Reduce[ops.AssociativeOp, Funsor, frozenset]", "Funsor"),
    ("Reduce[ops.AssociativeOp, Funsor, frozenset]", "Reduce"),
    ("Reduce[ops.AssociativeOp, Funsor, frozenset]", "Reduce[ops.Op, Funsor, frozenset]"),
    ("Reduce[ops.AssociativeOp, Reduce[ops.AssociativeOp, Funsor, frozenset], frozenset]",
     "Reduce[ops.Op, Funsor, frozenset]"),
    ("Reduce[ops.AssociativeOp, Reduce[ops.AssociativeOp, Funsor, frozenset], frozenset]",
     "Reduce[ops.AssociativeOp, Reduce, frozenset]"),
    ("Stack[str, typing.Tuple[Number, Number, Number]]", "Stack"),
    ("Stack[str, typing.Tuple[Number, Number, Number]]", "Stack[str, tuple]"),
    # Unions
    ("Reduce[ops.AssociativeOp, (Number, Stack[str, (tuple, typing.Tuple[Number, Number])]), frozenset]", "Funsor"),
    ("Reduce[ops.AssociativeOp, (Number, Stack), frozenset]", "Reduce[ops.Op, Funsor, frozenset]"),
    ("Reduce[ops.AssociativeOp, (Stack, Reduce[ops.AssociativeOp, (Number, Stack), frozenset]), frozenset]",
     "Reduce[(ops.Op, ops.AssociativeOp), Stack, frozenset]"),
])
def test_parametric_subclass(subcls_expr, cls_expr):
    subcls = eval(subcls_expr)
    cls = eval(cls_expr)
    print(subcls.classname)
    print(cls.classname)
    assert issubclass(cls, (Funsor, Reduce)) and not issubclass(subcls, typing.Tuple)  # appease flake8
    assert issubclass(subcls, cls)


@pytest.mark.parametrize("subcls_expr,cls_expr", [
    ("Funsor", "Reduce[ops.AssociativeOp, Funsor, frozenset]"),
    ("Reduce", "Reduce[ops.AssociativeOp, Funsor, frozenset]"),
    ("Reduce[ops.Op, Funsor, frozenset]", "Reduce[ops.AssociativeOp, Funsor, frozenset]"),
    ("Reduce[ops.AssociativeOp, Reduce[ops.AssociativeOp, Funsor, frozenset], frozenset]",
     "Reduce[ops.Op, Variable, frozenset]"),
    ("Reduce[ops.AssociativeOp, Reduce[ops.AssociativeOp, Funsor, frozenset], frozenset]",
     "Reduce[ops.AssociativeOp, Reduce[ops.AddOp, Funsor, frozenset], frozenset]"),
    ("Stack", "Stack[str, typing.Tuple[Number, Number, Number]]"),
    ("Stack[str, tuple]", "Stack[str, typing.Tuple[Number, Number, Number]]"),
    ("Stack[str, typing.Tuple[Number, Number]]", "Stack[str, typing.Tuple[Number, Reduce]]"),
    ("Stack[str, typing.Tuple[Number, Reduce]]", "Stack[str, typing.Tuple[Number, Number]]"),
    # Unions
    ("Funsor", "Reduce[ops.AssociativeOp, (Number, Funsor), frozenset]"),
    ("Reduce[ops.Op, Funsor, frozenset]", "Reduce[ops.AssociativeOp, (Number, Stack), frozenset]"),
    ("Reduce[(ops.Op, ops.AssociativeOp), Stack, frozenset]",
     "Reduce[ops.AssociativeOp, (Stack[str, tuple], Reduce[ops.AssociativeOp, (Cat, Stack), frozenset]), frozenset]"),
])
def test_not_parametric_subclass(subcls_expr, cls_expr):
    subcls = eval(subcls_expr)
    cls = eval(cls_expr)
    print(subcls.classname)
    print(cls.classname)
    assert issubclass(cls, (Funsor, Reduce)) and not issubclass(subcls, typing.Tuple)  # appease flake8
    assert not issubclass(subcls, cls)


@pytest.mark.parametrize("start,stop", [
    (1, 3), (0, 1), (3, 7), (4, 8), (0, 2), (0, 10), (1, 2), (1, 10), (2, 10)])
@pytest.mark.parametrize("step", [1, 2, 3, 5, 10])
def test_cat_slice_tensor(start, stop, step):

    terms = tuple(
        random_tensor(OrderedDict([('t', Bint[t]), ('a', Bint[2])]))
        for t in [2, 1, 3, 4, 1, 3])
    dtype = sum(term.inputs['t'].dtype for term in terms)
    sub = Slice('t', start, stop, step, dtype)

    # eager
    expected = Cat('t', terms)(t=sub)

    # lazy - exercise Cat.eager_subs
    with interpretation(lazy):
        actual = Cat('t', terms)(t=sub)
    actual = reinterpret(actual)

    assert_close(actual, expected)


@pytest.mark.parametrize("dtype", ["real", 2, 3])
def test_stack_lambda(dtype):

    x1 = Number(0, dtype)
    x2 = Number(1, dtype)

    y = Stack("i", (x1, x2))

    z = Lambda(Variable("i", Bint[2]), y)

    assert y.shape == ()
    assert z.output == Array[dtype, (2,)]

    assert z[0] is x1
    assert z[1] is x2


def test_funsor_tuple():
    x = Number(1, 3)
    y = Number(2.5, 'real')
    z = random_tensor(OrderedDict([('i', Bint[2])]))

    xyz = Tuple((x, y, z))

    check_funsor(xyz, {'i': Bint[2]}, Product[x.output, y.output, z.output])

    assert eval(repr(xyz.output)) is xyz.output

    assert xyz[0] is x
    assert xyz[1] is y
    assert xyz[2] is z

    x1, y1, z1 = xyz
    assert x1 is x
    assert y1 is y
    assert z1 is z
