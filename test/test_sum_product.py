# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import re
import os
from collections import OrderedDict
from functools import partial, reduce

import pytest

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer
from funsor.sum_product import (
    MarkovProduct,
    _partition,
    mixed_sequential_sum_product,
    naive_sarkka_bilmes_product,
    naive_sequential_sum_product,
    partial_sum_product,
    modified_partial_sum_product,
    sarkka_bilmes_product,
    sequential_sum_product,
    modified_sequential_sum_product,
    sum_product
)
from funsor.tensor import Tensor, get_default_prototype
from funsor.terms import Variable, eager_or_die, moment_matching, reflect
from funsor.testing import assert_close, random_gaussian, random_tensor
from funsor.util import get_backend

pytestmark = pytest.mark.skipif((get_backend() == 'jax') and ('CI' in os.environ), reason='slow tests')


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
    terms = [random_tensor(OrderedDict((s, Bint[sizes[s]]) for s in input_))
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
@pytest.mark.parametrize('impl', [
    partial_sum_product,
    modified_partial_sum_product,
])
def test_partial_sum_product(impl, sum_op, prod_op, inputs, plates, vars1, vars2):
    inputs = inputs.split(',')
    factors = [random_tensor(OrderedDict((d, Bint[2]) for d in ds)) for ds in inputs]
    plates = frozenset(plates)
    vars1 = frozenset(vars1)
    vars2 = frozenset(vars2)

    factors1 = impl(sum_op, prod_op, factors, vars1, plates)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plates)
    actual = reduce(prod_op, factors2)

    expected = sum_product(sum_op, prod_op, factors, vars1 | vars2, plates)
    assert_close(actual, expected)


@pytest.mark.parametrize('sum_op,prod_op', [(ops.logaddexp, ops.add), (ops.add, ops.mul)])
@pytest.mark.parametrize('inputs,plates,time,step', [('xn,xpcnt,cynti,cyzntij', 'nij', 't', {'p': 'c'})])
@pytest.mark.parametrize('vars1,vars2', [
    ('', 'xpcyzntij'),
    ('zj', 'xpcynti'),
    ('yzij', 'xpcnt'),
    ('pcyztij', 'xn'),
    ('xpcyztij', 'n'),
])
def test_markov_partial_sum_product(sum_op, prod_op, inputs, plates, time, step, vars1, vars2):
    inputs = inputs.split(',')
    factors = [random_tensor(OrderedDict((d, Bint[2]) if d != 't' else (d, Bint[10]) for d in ds)) for ds in inputs]
    plates = frozenset(plates)
    vars1 = frozenset(vars1)
    vars2 = frozenset(vars2)

    factors1 = modified_partial_sum_product(sum_op, prod_op, factors, vars1, plates, time, step)
    factors2 = modified_partial_sum_product(sum_op, prod_op, factors1, vars2, plates, time, step)
    actual = reduce(prod_op, factors2)

    unrolled_factors = []
    unrolled_plates = frozenset({'n'})
    for f in factors:
        if time in f.inputs:
            for t in range(f.inputs[time].size):
                arg_to_time = {}
                for arg in f.inputs:
                    if arg == time:
                        arg_to_time[arg] = t
                    elif arg in ['x', 'n']:
                        pass
                    elif arg in step.keys():
                        arg_to_time[arg] = '{}_{}'.format(step[arg], t-1)
                    else:
                        arg_to_time[arg] = '{}_{}'.format(arg, t)
                        if arg in plates:
                            unrolled_plates |= frozenset({arg_to_time[arg]})
                unrolled_factors.append(f(**arg_to_time))
        else:
            unrolled_factors.append(f)
    unrolled_vars = frozenset()
    for f in unrolled_factors:
        unrolled_vars |= frozenset(f.inputs)
    expected = sum_product(sum_op, prod_op, unrolled_factors, unrolled_vars, unrolled_plates)

    assert_close(actual, expected)


@pytest.mark.parametrize('sum_op,prod_op', [(ops.logaddexp, ops.add), (ops.add, ops.mul)])
@pytest.mark.parametrize('inputs,plates,time,step', [('x,xpct,crst,csyti,cyztij', 'ij', 't', {'p': 'c', 'r': 's'})])
@pytest.mark.parametrize('vars1,vars2', [
    ('', 'xpcrsyztij'),
    ('zj', 'xpcrsyti'),
    ('yzij', 'xpcrst'),
    ('pcrsyztij', 'x'),
    ('xpcrsyztij', ''),
])
def test_markov_partial_sum_product_multi(sum_op, prod_op, inputs, plates, time, step, vars1, vars2):
    inputs = inputs.split(',')
    factors = [random_tensor(OrderedDict((d, Bint[2]) if d != 't' else (d, Bint[10]) for d in ds)) for ds in inputs]
    plates = frozenset(plates)
    vars1 = frozenset(vars1)
    vars2 = frozenset(vars2)

    factors1 = modified_partial_sum_product(sum_op, prod_op, factors, vars1, plates, time, step)
    factors2 = modified_partial_sum_product(sum_op, prod_op, factors1, vars2, plates, time, step)
    actual = reduce(prod_op, factors2)

    unrolled_factors = []
    unrolled_plates = frozenset()
    for f in factors:
        if time in f.inputs:
            for t in range(f.inputs[time].size):
                arg_to_time = {}
                for arg in f.inputs:
                    if arg == time:
                        arg_to_time[arg] = t
                    elif arg == 'x':
                        pass
                    elif arg in step.keys():
                        arg_to_time[arg] = '{}_{}'.format(step[arg], t-1)
                    else:
                        arg_to_time[arg] = '{}_{}'.format(arg, t)
                        if arg in plates:
                            unrolled_plates |= frozenset({arg_to_time[arg]})
                unrolled_factors.append(f(**arg_to_time))
        else:
            unrolled_factors.append(f)

    unrolled_vars = frozenset()
    for f in unrolled_factors:
        unrolled_vars |= frozenset(f.inputs)
    expected = sum_product(sum_op, prod_op, unrolled_factors, unrolled_vars, unrolled_plates)

    assert_close(actual, expected)


@pytest.mark.parametrize('num_steps', [None] + list(range(1, 13)))
@pytest.mark.parametrize('sum_op,prod_op,state_domain', [
    (ops.add, ops.mul, Bint[2]),
    (ops.add, ops.mul, Bint[3]),
    (ops.logaddexp, ops.add, Bint[2]),
    (ops.logaddexp, ops.add, Bint[3]),
    (ops.logaddexp, ops.add, Real),
    (ops.logaddexp, ops.add, Reals[2]),
], ids=str)
@pytest.mark.parametrize('batch_inputs', [
    OrderedDict(),
    OrderedDict([("foo", Bint[5])]),
    OrderedDict([("foo", Bint[2]), ("bar", Bint[4])]),
], ids=lambda d: ",".join(d.keys()))
@pytest.mark.parametrize('impl', [
    sequential_sum_product,
    naive_sequential_sum_product,
    modified_sequential_sum_product,
    MarkovProduct,
    partial(mixed_sequential_sum_product, num_segments=2),
    partial(mixed_sequential_sum_product, num_segments=3),
])
def test_sequential_sum_product(impl, sum_op, prod_op, batch_inputs, state_domain, num_steps):
    inputs = OrderedDict(batch_inputs)
    inputs.update(prev=state_domain, curr=state_domain)
    if num_steps is None:
        num_steps = 1
    else:
        inputs["time"] = Bint[num_steps]
    if state_domain.dtype == "real":
        trans = random_gaussian(inputs)
    else:
        trans = random_tensor(inputs)
    time = Variable("time", Bint[num_steps])

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
    OrderedDict(),
    OrderedDict([("foo", Bint[5])]),
    OrderedDict([("foo", Bint[2]), ("bar", Bint[4])]),
], ids=lambda d: ",".join(d.keys()))
@pytest.mark.parametrize('x_domain,y_domain', [
    (Bint[2], Bint[3]),
    (Real, Reals[2, 2]),
    (Bint[2], Reals[2]),
], ids=str)
@pytest.mark.parametrize('impl', [
    sequential_sum_product,
    naive_sequential_sum_product,
    modified_sequential_sum_product,
    MarkovProduct,
    partial(mixed_sequential_sum_product, num_segments=2),
    partial(mixed_sequential_sum_product, num_segments=3),
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
        inputs["time"] = Bint[num_steps]
    if any(v.dtype == "real" for v in inputs.values()):
        trans = random_gaussian(inputs)
    else:
        trans = random_tensor(inputs)
    time = Variable("time", Bint[num_steps])
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
@pytest.mark.parametrize('impl', [
    sequential_sum_product,
    modified_sequential_sum_product,
])
def test_sequential_sum_product_bias_1(impl, num_steps, dim):
    time = Variable("time", Bint[num_steps])
    bias_dist = random_gaussian(OrderedDict([
        ("bias", Reals[dim]),
    ]))
    trans = random_gaussian(OrderedDict([
        ("time", Bint[num_steps]),
        ("x_prev", Reals[dim]),
        ("x_curr", Reals[dim]),
    ]))
    obs = random_gaussian(OrderedDict([
        ("time", Bint[num_steps]),
        ("x_curr", Reals[dim]),
        ("bias", Reals[dim]),
    ]))
    factor = trans + obs + bias_dist
    assert set(factor.inputs) == {"time", "bias", "x_prev", "x_curr"}

    result = impl(ops.logaddexp, ops.add, factor, time, {"x_prev": "x_curr"})
    assert set(result.inputs) == {"bias", "x_prev", "x_curr"}


@pytest.mark.xfail(reason="missing pattern for Gaussian(x=y[z]) for real x")
@pytest.mark.parametrize("num_steps", [1, 2, 3, 10])
@pytest.mark.parametrize("num_sensors", [2])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize('impl', [
    sequential_sum_product,
    modified_sequential_sum_product,
])
def test_sequential_sum_product_bias_2(impl, num_steps, num_sensors, dim):
    time = Variable("time", Bint[num_steps])
    bias = Variable("bias", Reals[num_sensors, dim])
    bias_dist = random_gaussian(OrderedDict([
        ("bias", Reals[num_sensors, dim]),
    ]))
    trans = random_gaussian(OrderedDict([
        ("time", Bint[num_steps]),
        ("x_prev", Reals[dim]),
        ("x_curr", Reals[dim]),
    ]))
    obs = random_gaussian(OrderedDict([
        ("time", Bint[num_steps]),
        ("x_curr", Reals[dim]),
        ("bias", Reals[dim]),
    ]))

    # Each time step only a single sensor observes x,
    # and each sensor has a different bias.
    sensor_id = Tensor(ops.new_arange(get_default_prototype(), num_steps) % 2,
                       OrderedDict(time=Bint[num_steps]), dtype=2)
    with interpretation(eager_or_die):
        factor = trans + obs(bias=bias[sensor_id]) + bias_dist
    assert set(factor.inputs) == {"time", "bias", "x_prev", "x_curr"}

    result = impl(ops.logaddexp, ops.add, factor, time, {"x_prev": "x_curr"})
    assert set(result.inputs) == {"bias", "x_prev", "x_curr"}


def _check_sarkka_bilmes(trans, expected_inputs, global_vars, num_periods=1):

    sum_op, prod_op = ops.logaddexp, ops.add

    assert "time" in trans.inputs
    duration = trans.inputs["time"].dtype
    time_var = Variable("time", Bint[duration])

    expected = naive_sarkka_bilmes_product(sum_op, prod_op, trans, time_var, global_vars)
    assert dict(expected.inputs) == expected_inputs

    actual = sarkka_bilmes_product(sum_op, prod_op, trans, time_var, global_vars,
                                   num_periods=num_periods)
    assert dict(actual.inputs) == expected_inputs

    actual = actual.align(tuple(expected.inputs.keys()))
    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def _check_modified_sequential(trans, expected_inputs, global_vars, step, window):

    sum_op, prod_op = ops.logaddexp, ops.add

    assert "time" in trans.inputs
    duration = trans.inputs["time"].dtype
    time_var = Variable("time", Bint[duration])

    actual = modified_sequential_sum_product(sum_op, prod_op, trans, time_var, step, window)

    expected = naive_sarkka_bilmes_product(sum_op, prod_op, trans, time_var, global_vars)
    assert dict(expected.inputs) == expected_inputs

    actual = actual.align(tuple(expected.inputs.keys()))
    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6])
def test_sarkka_bilmes_example_0(duration):

    trans = random_tensor(OrderedDict({
        "time": Bint[duration],
        "a": Bint[3],
    }))

    expected_inputs = {
        "a": Bint[3],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6])
def test_sarkka_bilmes_example_1(duration):

    trans = random_tensor(OrderedDict({
        "time": Bint[duration],
        "a": Bint[3],
        "b": Bint[2],
        "Pb": Bint[2],
    }))

    expected_inputs = {
        "a": Bint[3],
        "b": Bint[2],
        "Pb": Bint[2],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())

    step = (('Pa', 'a'), ('Pb', 'b'))
    _check_modified_sequential(trans, expected_inputs, frozenset(), step, 1)


@pytest.mark.parametrize("duration", [2, 4, 6, 8])
def test_sarkka_bilmes_example_2(duration):

    trans = random_tensor(OrderedDict({
        "time": Bint[duration],
        "a": Bint[4],
        "b": Bint[3],
        "Pb": Bint[3],
        "c": Bint[2],
        "PPc": Bint[2],
    }))

    expected_inputs = {
        "a": Bint[4],
        "b": Bint[3],
        "Pb": Bint[3],
        "c": Bint[2],
        "PPc": Bint[2],
        "Pc": Bint[2],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())

    step = (('PPa', 'Pa', 'a'), ('PPb', 'Pb', 'b'), ('PPc', 'Pc', 'c'))
    _check_modified_sequential(trans, expected_inputs, frozenset(), step, 2)


@pytest.mark.parametrize("duration", [2, 4, 6, 8])
def test_sarkka_bilmes_example_3(duration):

    trans = random_tensor(OrderedDict({
        "time": Bint[duration],
        "a": Bint[4],
        "c": Bint[2],
        "PPc": Bint[2],
    }))

    expected_inputs = {
        "a": Bint[4],
        "c": Bint[2],
        "PPc": Bint[2],
        "Pc": Bint[2],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())

    step = (('PPa', 'Pa', 'a'), ('PPc', 'Pc', 'c'))
    _check_modified_sequential(trans, expected_inputs, frozenset(), step, 2)


@pytest.mark.parametrize("duration", [3, 6, 9])
def test_sarkka_bilmes_example_4(duration):

    trans = random_tensor(OrderedDict({
        "time": Bint[duration],
        "a": Bint[2],
        "Pa": Bint[2],
        "PPPa": Bint[2],
    }))

    expected_inputs = {
        "a": Bint[2],
        "PPa": Bint[2],
        "PPPa": Bint[2],
        "Pa": Bint[2],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())

    step = (('PPPa', 'PPa', 'Pa', 'a'),)
    _check_modified_sequential(trans, expected_inputs, frozenset(), step, 3)


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6])
def test_sarkka_bilmes_example_5(duration):

    trans = random_tensor(OrderedDict({
        "time": Bint[duration],
        "a": Bint[3],
        "Pa": Bint[3],
        "x": Bint[2],
    }))

    expected_inputs = {
        "a": Bint[3],
        "Pa": Bint[3],
        "x": Bint[2],
    }

    global_vars = frozenset(["x"])

    _check_sarkka_bilmes(trans, expected_inputs, global_vars)

    step = (('Pa', 'a'),)
    _check_modified_sequential(trans, expected_inputs, global_vars, step, 1)


@pytest.mark.parametrize("duration", [3, 6, 9])
def test_sarkka_bilmes_example_6(duration):

    trans = random_tensor(OrderedDict({
        "time": Bint[duration],
        "a": Bint[2],
        "Pa": Bint[2],
        "PPPa": Bint[2],
        "x": Bint[3],
    }))

    expected_inputs = {
        "a": Bint[2],
        "PPa": Bint[2],
        "PPPa": Bint[2],
        "Pa": Bint[2],
        "x": Bint[3],
    }

    global_vars = frozenset(["x"])

    _check_sarkka_bilmes(trans, expected_inputs, global_vars)

    step = (('PPPa', 'PPa', 'Pa', 'a'),)
    _check_modified_sequential(trans, expected_inputs, global_vars, step, 3)


@pytest.mark.parametrize("time_input", [("time", Bint[t]) for t in range(2, 10)])
@pytest.mark.parametrize("global_inputs", [
    (),
    (("x", Bint[2]),),
])
@pytest.mark.parametrize("local_inputs", [
    # tensor
    (("a", Bint[2]),),
    (("a", Bint[2]), ("Pa", Bint[2])),
    (("a", Bint[2]), ("b", Bint[2]), ("Pb", Bint[2])),
    (("a", Bint[2]), ("b", Bint[2]), ("PPb", Bint[2])),
    (("a", Bint[2]), ("b", Bint[2]), ("Pb", Bint[2]), ("c", Bint[2]), ("PPc", Bint[2])),
    (("a", Bint[2]), ("Pa", Bint[2]), ("PPPa", Bint[2])),
    (("a", Bint[2]), ("b", Bint[2]), ("PPb", Bint[2]), ("PPPa", Bint[2])),
    # gaussian
    (("a", Real),),
    (("a", Real), ("Pa", Real)),
    (("a", Real), ("b", Real), ("Pb", Real)),
    (("a", Real), ("b", Real), ("PPb", Real)),
    (("a", Real), ("b", Real), ("Pb", Real), ("c", Real), ("PPc", Real)),
    (("a", Real), ("Pa", Real), ("PPPa", Real)),
    (("a", Real), ("b", Real), ("PPb", Real), ("PPPa", Real)),
    # mv gaussian
    (("a", Reals[2]), ("b", Reals[2]), ("Pb", Reals[2])),
    (("a", Reals[2]), ("b", Reals[2]), ("PPb", Reals[2])),
])
@pytest.mark.parametrize("num_periods", [1, 2])
def test_sarkka_bilmes_generic(time_input, global_inputs, local_inputs, num_periods):

    lags = {
        kk: reduce(max, [
            len(re.search("^P*", k).group(0)) for k, v in local_inputs
            if k.strip("P") == kk], 0)
        for kk, vv in local_inputs if not kk.startswith("P")
    }
    expected_inputs = dict(global_inputs + tuple(set(
        ((t * "P" + k), v)
        for k, v in local_inputs if not k.startswith("P")
        for t in range(0, lags[k] + 1))))

    trans_inputs = OrderedDict(global_inputs + (time_input,) + local_inputs)
    global_vars = frozenset(k for k, v in global_inputs)

    if any(v.dtype == "real" for v in trans_inputs.values()):
        trans = random_gaussian(trans_inputs)
    else:
        trans = random_tensor(trans_inputs)

    try:
        _check_sarkka_bilmes(trans, expected_inputs, global_vars, num_periods)
    except NotImplementedError as e:
        partial_reasons = (
            'TODO handle partial windows',
        )
        if any(reason in e.args[0] for reason in partial_reasons):
            pytest.xfail(reason=e.args[0])
        else:
            raise


@pytest.mark.parametrize("duration,num_segments", [(12, 1), (12, 2), (12, 3), (12, 4), (12, 6)])
def test_mixed_sequential_sum_product(duration, num_segments):

    sum_op, prod_op = ops.logaddexp, ops.add
    time_var = Variable("time", Bint[duration])
    step = {"Px": "x"}

    trans_inputs = ((time_var.name, Bint[duration]),) + \
        tuple((k, Bint[2]) for k in step.keys()) + \
        tuple((v, Bint[2]) for v in step.values())

    trans = random_tensor(OrderedDict(trans_inputs))

    expected = sequential_sum_product(sum_op, prod_op, trans, time_var, step)
    actual = mixed_sequential_sum_product(sum_op, prod_op, trans, time_var, step,
                                          num_segments=num_segments)

    assert_close(actual, expected)
