from collections import OrderedDict
from functools import reduce

import pytest
import torch

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer
from funsor.sum_product import (
    MarkovProduct,
    _partition,
    naive_sarkka_bilmes_product,
    naive_sequential_sum_product,
    partial_sum_product,
    sarkka_bilmes_product,
    sequential_sum_product,
    sum_product
)
from funsor.terms import Variable, eager_or_die, moment_matching, reflect
from funsor.testing import assert_close, random_gaussian, random_tensor
from funsor.torch import Tensor


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


@pytest.mark.parametrize('sum_op,prod_op', [(ops.add, ops.mul), (ops.logaddexp, ops.add)])
@pytest.mark.parametrize('inputs,plates', [('a,abi,bcij', 'ij')])
@pytest.mark.parametrize('vars1,vars2', [
    ('', 'abcij'),
    ('c', 'abij'),
    ('cj', 'abi'),
    ('bcj', 'ai'),
    ('bcij', 'a'),
    ('abcij', ''),
])
def test_partial_sum_product(sum_op, prod_op, inputs, plates, vars1, vars2):
    inputs = inputs.split(',')
    factors = [random_tensor(OrderedDict((d, bint(2)) for d in ds)) for ds in inputs]
    plates = frozenset(plates)
    vars1 = frozenset(vars1)
    vars2 = frozenset(vars2)

    factors1 = partial_sum_product(sum_op, prod_op, factors, vars1, plates)
    factors2 = partial_sum_product(sum_op, prod_op, factors1, vars2, plates)
    actual = reduce(prod_op, factors2)

    expected = sum_product(sum_op, prod_op, factors, vars1 | vars2, plates)
    assert_close(actual, expected)


@pytest.mark.parametrize('num_steps', [None] + list(range(1, 13)))
@pytest.mark.parametrize('sum_op,prod_op,state_domain', [
    (ops.add, ops.mul, bint(2)),
    (ops.add, ops.mul, bint(3)),
    (ops.logaddexp, ops.add, bint(2)),
    (ops.logaddexp, ops.add, bint(3)),
    (ops.logaddexp, ops.add, reals()),
    (ops.logaddexp, ops.add, reals(2)),
], ids=str)
@pytest.mark.parametrize('batch_inputs', [
    {},
    {"foo": bint(5)},
    {"foo": bint(2), "bar": bint(4)},
], ids=lambda d: ",".join(d.keys()))
@pytest.mark.parametrize('impl', [
    sequential_sum_product,
    naive_sequential_sum_product,
    MarkovProduct,
])
def test_sequential_sum_product(impl, sum_op, prod_op, batch_inputs, state_domain, num_steps):
    inputs = OrderedDict(batch_inputs)
    inputs.update(prev=state_domain, curr=state_domain)
    if num_steps is None:
        num_steps = 1
    else:
        inputs["time"] = bint(num_steps)
    if state_domain.dtype == "real":
        trans = random_gaussian(inputs)
    else:
        trans = random_tensor(inputs)
    time = Variable("time", bint(num_steps))

    actual = impl(sum_op, prod_op, trans, time, {"prev": "curr"})
    expected_inputs = batch_inputs.copy()
    expected_inputs.update(prev=state_domain, curr=state_domain)
    assert dict(actual.inputs) == expected_inputs

    # Check against contract.
    operands = tuple(trans(time=t, prev="t_{}".format(t), curr="t_{}".format(t+1))
                     for t in range(num_steps))
    reduce_vars = frozenset("t_{}".format(t) for t in range(1, num_steps))
    with interpretation(reflect):
        expected = sum_product(sum_op, prod_op, operands, reduce_vars)
    expected = apply_optimizer(expected)
    expected = expected(**{"t_0": "prev", "t_{}".format(num_steps): "curr"})
    expected = expected.align(tuple(actual.inputs.keys()))
    assert_close(actual, expected, rtol=5e-4 * num_steps)


@pytest.mark.parametrize('num_steps', [None] + list(range(1, 6)))
@pytest.mark.parametrize('batch_inputs', [
    {},
    {"foo": bint(5)},
    {"foo": bint(2), "bar": bint(4)},
], ids=lambda d: ",".join(d.keys()))
@pytest.mark.parametrize('x_domain,y_domain', [
    (bint(2), bint(3)),
    (reals(), reals(2, 2)),
    (bint(2), reals(2)),
], ids=str)
@pytest.mark.parametrize('impl', [
    sequential_sum_product,
    naive_sequential_sum_product,
    MarkovProduct,
])
def test_sequential_sum_product_multi(impl, x_domain, y_domain, batch_inputs, num_steps):
    sum_op = ops.logaddexp
    prod_op = ops.add
    inputs = OrderedDict(batch_inputs)
    inputs.update(x_prev=x_domain, x_curr=x_domain,
                  y_prev=y_domain, y_curr=y_domain)
    if num_steps is None:
        num_steps = 1
    else:
        inputs["time"] = bint(num_steps)
    if any(v.dtype == "real" for v in inputs.values()):
        trans = random_gaussian(inputs)
    else:
        trans = random_tensor(inputs)
    time = Variable("time", bint(num_steps))
    step = {"x_prev": "x_curr", "y_prev": "y_curr"}

    with interpretation(moment_matching):
        actual = impl(sum_op, prod_op, trans, time, step)
        expected_inputs = batch_inputs.copy()
        expected_inputs.update(x_prev=x_domain, x_curr=x_domain,
                               y_prev=y_domain, y_curr=y_domain)
        assert dict(actual.inputs) == expected_inputs

        # Check against contract.
        operands = tuple(trans(time=t,
                               x_prev="x_{}".format(t), x_curr="x_{}".format(t+1),
                               y_prev="y_{}".format(t), y_curr="y_{}".format(t+1))
                         for t in range(num_steps))
        reduce_vars = frozenset("x_{}".format(t) for t in range(1, num_steps)).union(
                                "y_{}".format(t) for t in range(1, num_steps))
        expected = sum_product(sum_op, prod_op, operands, reduce_vars)
        expected = expected(**{"x_0": "x_prev", "x_{}".format(num_steps): "x_curr",
                               "y_0": "y_prev", "y_{}".format(num_steps): "y_curr"})
        expected = expected.align(tuple(actual.inputs.keys()))


@pytest.mark.parametrize("num_steps", [1, 2, 3, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sequential_sum_product_bias_1(num_steps, dim):
    time = Variable("time", bint(num_steps))
    bias_dist = random_gaussian(OrderedDict([
        ("bias", reals(dim)),
    ]))
    trans = random_gaussian(OrderedDict([
        ("time", bint(num_steps)),
        ("x_prev", reals(dim)),
        ("x_curr", reals(dim)),
    ]))
    obs = random_gaussian(OrderedDict([
        ("time", bint(num_steps)),
        ("x_curr", reals(dim)),
        ("bias", reals(dim)),
    ]))
    factor = trans + obs + bias_dist
    assert set(factor.inputs) == {"time", "bias", "x_prev", "x_curr"}

    result = sequential_sum_product(ops.logaddexp, ops.add, factor, time, {"x_prev": "x_curr"})
    assert set(result.inputs) == {"bias", "x_prev", "x_curr"}


@pytest.mark.xfail(reason="missing pattern for Gaussian(x=y[z]) for real x")
@pytest.mark.parametrize("num_steps", [1, 2, 3, 10])
@pytest.mark.parametrize("num_sensors", [2])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sequential_sum_product_bias_2(num_steps, num_sensors, dim):
    time = Variable("time", bint(num_steps))
    bias = Variable("bias", reals(num_sensors, dim))
    bias_dist = random_gaussian(OrderedDict([
        ("bias", reals(num_sensors, dim)),
    ]))
    trans = random_gaussian(OrderedDict([
        ("time", bint(num_steps)),
        ("x_prev", reals(dim)),
        ("x_curr", reals(dim)),
    ]))
    obs = random_gaussian(OrderedDict([
        ("time", bint(num_steps)),
        ("x_curr", reals(dim)),
        ("bias", reals(dim)),
    ]))

    # Each time step only a single sensor observes x,
    # and each sensor has a different bias.
    sensor_id = Tensor(torch.arange(num_steps) % 2, OrderedDict(time=bint(num_steps)), dtype=2)
    with interpretation(eager_or_die):
        factor = trans + obs(bias=bias[sensor_id]) + bias_dist
    assert set(factor.inputs) == {"time", "bias", "x_prev", "x_curr"}

    result = sequential_sum_product(ops.logaddexp, ops.add, factor, time, {"x_prev": "x_curr"})
    assert set(result.inputs) == {"bias", "x_prev", "x_curr"}


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6])
def test_sarkka_bilmes_example_1(duration):

    sum_op, prod_op = ops.logaddexp, ops.add

    time_var = Variable("time", bint(duration))

    trans = random_tensor(OrderedDict({
        "time": bint(duration),
        "a": bint(3),
        "b": bint(2),
        "Pb": bint(2),
    }))

    expected_inputs = {
        "a": bint(3),
        "b": bint(2),
        "Pb": bint(2),
    }

    print("\n")
    expected = naive_sarkka_bilmes_product(sum_op, prod_op, trans, time_var)
    assert dict(expected.inputs) == expected_inputs

    actual = sarkka_bilmes_product(sum_op, prod_op, trans, time_var)
    assert dict(actual.inputs) == expected_inputs

    actual = actual.align(tuple(expected.inputs.keys()))
    assert_close(actual, expected)


@pytest.mark.parametrize("duration", [2, 4, 6, 8])
def test_sarkka_bilmes_example_2(duration):

    sum_op, prod_op = ops.logaddexp, ops.add

    time_var = Variable("time", bint(duration))

    trans = random_tensor(OrderedDict({
        "time": bint(duration),
        "a": bint(4),
        "b": bint(3),
        "Pb": bint(3),
        "c": bint(2),
        "PPc": bint(2),
    }))

    expected_inputs = {
        "a": bint(4),
        "b": bint(3),
        "Pb": bint(3),
        "c": bint(2),
        "PPc": bint(2),
        "Pc": bint(2),
    }

    print("\n")
    expected = naive_sarkka_bilmes_product(sum_op, prod_op, trans, time_var)
    assert dict(expected.inputs) == expected_inputs

    actual = sarkka_bilmes_product(sum_op, prod_op, trans, time_var)
    assert dict(actual.inputs) == expected_inputs

    actual = actual.align(tuple(expected.inputs.keys()))
    assert_close(actual, expected)
