# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from collections import OrderedDict
from functools import partial, reduce

import pytest

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.interpretations import (
    compress_gaussians,
    eager,
    eager_or_die,
    lazy,
    moment_matching,
    reflect,
)
from funsor.interpreter import reinterpret
from funsor.optimizer import apply_optimizer
from funsor.sum_product import (
    MarkovProduct,
    _partition,
    dynamic_partial_sum_product,
    mixed_sequential_sum_product,
    modified_partial_sum_product,
    naive_sarkka_bilmes_product,
    naive_sequential_sum_product,
    partial_sum_product,
    partial_unroll,
    sarkka_bilmes_product,
    sequential_sum_product,
    sum_product,
)
from funsor.tensor import Tensor, get_default_prototype
from funsor.terms import Variable
from funsor.testing import assert_close, random_gaussian, random_tensor
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(
    (get_backend() == "jax") and ("CI" in os.environ), reason="slow tests"
)


@pytest.mark.parametrize(
    "inputs,dims,expected_num_components",
    [
        ([""], set(), 1),
        (["a"], set(), 1),
        (["a"], set("a"), 1),
        (["a", "a"], set(), 2),
        (["a", "a"], set("a"), 1),
        (["a", "a", "b", "b"], set(), 4),
        (["a", "a", "b", "b"], set("a"), 3),
        (["a", "a", "b", "b"], set("b"), 3),
        (["a", "a", "b", "b"], set("ab"), 2),
        (["a", "ab", "b"], set(), 3),
        (["a", "ab", "b"], set("a"), 2),
        (["a", "ab", "b"], set("b"), 2),
        (["a", "ab", "b"], set("ab"), 1),
        (["a", "ab", "bc", "c"], set(), 4),
        (["a", "ab", "bc", "c"], set("c"), 3),
        (["a", "ab", "bc", "c"], set("b"), 3),
        (["a", "ab", "bc", "c"], set("a"), 3),
        (["a", "ab", "bc", "c"], set("ac"), 2),
        (["a", "ab", "bc", "c"], set("abc"), 1),
    ],
)
def test_partition(inputs, dims, expected_num_components):
    sizes = dict(zip("abc", [2, 3, 4]))
    terms = [
        random_tensor(OrderedDict((s, Bint[sizes[s]]) for s in input_))
        for input_ in inputs
    ]
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


@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.add, ops.mul), (ops.logaddexp, ops.add)]
)
@pytest.mark.parametrize("inputs,plates", [("a,abi,bcij", "ij")])
@pytest.mark.parametrize(
    "vars1,vars2",
    [
        ("", "abcij"),
        ("c", "abij"),
        ("cj", "abi"),
        ("bcj", "ai"),
        ("bcij", "a"),
        ("abcij", ""),
    ],
)
@pytest.mark.parametrize(
    "impl",
    [partial_sum_product, modified_partial_sum_product, dynamic_partial_sum_product],
)
def test_partial_sum_product(impl, sum_op, prod_op, inputs, plates, vars1, vars2):
    inputs = inputs.split(",")
    factors = [random_tensor(OrderedDict((d, Bint[2]) for d in ds)) for ds in inputs]
    vars1 = frozenset(vars1)
    vars2 = frozenset(vars2)

    plate_to_step = {k: {} for k in plates}
    if impl is partial_sum_product:
        plates = frozenset(plates)
    else:
        plates = {k: {} for k in plates}

    factors1 = impl(sum_op, prod_op, factors, vars1, plates)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plates)
    actual = reduce(prod_op, factors2)

    expected = sum_product(sum_op, prod_op, factors, vars1 | vars2, frozenset(plates))
    assert_close(actual, expected)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        unrolled_expected = reduce(prod_op, unrolled_factors).reduce(
            sum_op, unrolled_vars
        )
    unrolled_expected = apply_optimizer(unrolled_expected)
    assert_close(actual, unrolled_expected)


def test_partial_sum_product_batch_1():
    factor = random_gaussian(OrderedDict(i=Bint[2], j=Bint[2], x=Real))
    actual = sum_product(
        ops.logaddexp,
        ops.add,
        [factor],
        eliminate=frozenset("ix"),
        plates=frozenset("ij"),
    )
    assert actual.inputs == OrderedDict(j=Bint[2])


def test_partial_sum_product_batch_2():
    factor_i = random_gaussian(OrderedDict(i=Bint[2], x=Real))
    factor_ik = random_gaussian(OrderedDict(i=Bint[2], k=Bint[2], x=Real))
    actual = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_i, factor_ik],
        eliminate=frozenset("ix"),
        plates=frozenset("ik"),
    )
    assert actual.inputs == OrderedDict(k=Bint[2])


def test_intractable_1():
    factor_i = random_gaussian(OrderedDict(i=Bint[2], x=Real))
    factor_j = random_gaussian(OrderedDict(j=Bint[2], y=Real))
    factor_ij = random_gaussian(OrderedDict(i=Bint[2], j=Bint[2], x=Real, y=Real))
    actual = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_i, factor_j, factor_ij],
        eliminate=frozenset("ijxy"),
        plates=frozenset("ij"),
    )
    assert isinstance(actual, Tensor)
    assert not actual.inputs

    # Manually unroll plate j.
    factor_j0 = factor_j(j=0, y="y0")
    factor_j1 = factor_j(j=1, y="y1")
    factor_ij0 = factor_ij(j=0, y="y0")
    factor_ij1 = factor_ij(j=1, y="y1")
    expected = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_i, factor_j0, factor_j1, factor_ij0, factor_ij1],
        eliminate=frozenset(["i", "x", "y0", "y1"]),
        plates=frozenset("i"),
    )

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def test_intractable_2():
    factor_i = random_gaussian(OrderedDict(i=Bint[2], x=Real, zx=Real))
    factor_j = random_gaussian(OrderedDict(j=Bint[2], y=Real, zy=Real))
    factor_ij = random_gaussian(
        OrderedDict(i=Bint[2], j=Bint[2], x=Real, y=Real, zx=Real, zy=Real)
    )
    actual = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_i, factor_j, factor_ij],
        eliminate=frozenset(["i", "j", "x", "y", "zx", "zy"]),
        plates=frozenset("ij"),
    )
    assert isinstance(actual, Tensor)
    assert not actual.inputs

    # Manually unroll plate j.
    factor_j0 = factor_j(j=0, y="y0", zy="zy0")
    factor_j1 = factor_j(j=1, y="y1", zy="zy1")
    factor_ij0 = factor_ij(j=0, y="y0", zy="zy0")
    factor_ij1 = factor_ij(j=1, y="y1", zy="zy1")
    expected = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_i, factor_j0, factor_j1, factor_ij0, factor_ij1],
        eliminate=frozenset(["i", "x", "y0", "y1", "zx", "zy0", "zy1"]),
        plates=frozenset("i"),
    )

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def test_intractable_3():
    factor_ = random_gaussian(OrderedDict(w=Real))
    factor_i = random_gaussian(OrderedDict(i=Bint[2], w=Real, x=Real))
    factor_j = random_gaussian(OrderedDict(j=Bint[2], y=Real))
    factor_ij = random_gaussian(OrderedDict(i=Bint[2], j=Bint[2], x=Real, y=Real))
    actual = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_, factor_i, factor_j, factor_ij],
        eliminate=frozenset("ijxy"),
        plates=frozenset("ij"),
    )
    assert set(actual.inputs) == {"w"}

    # Manually unroll plate j.
    factor_j0 = factor_j(j=0, y="y0")
    factor_j1 = factor_j(j=1, y="y1")
    factor_ij0 = factor_ij(j=0, y="y0")
    factor_ij1 = factor_ij(j=1, y="y1")
    expected = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_, factor_i, factor_j0, factor_j1, factor_ij0, factor_ij1],
        eliminate=frozenset(["i", "x", "y0", "y1"]),
        plates=frozenset("i"),
    )
    assert set(expected.inputs) == {"w"}

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def test_intractable_4():
    factor_i = random_gaussian(OrderedDict(i=Bint[2], x=Real))
    factor_jk = random_gaussian(OrderedDict(j=Bint[2], k=Bint[2], y=Real))
    factor_ij = random_gaussian(OrderedDict(i=Bint[2], j=Bint[2], x=Real, y=Real))
    actual = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_i, factor_jk, factor_ij],
        eliminate=frozenset("ijxy"),
        plates=frozenset("ijk"),
    )
    assert set(actual.inputs) == {"k"}

    # Manually unroll plate j.
    factor_jk0 = factor_jk(j=0, y="y0")
    factor_jk1 = factor_jk(j=1, y="y1")
    factor_ij0 = factor_ij(j=0, y="y0")
    factor_ij1 = factor_ij(j=1, y="y1")
    expected = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_i, factor_jk0, factor_jk1, factor_ij0, factor_ij1],
        eliminate=frozenset(["i", "x", "y0", "y1"]),
        plates=frozenset("ik"),
    )
    assert set(expected.inputs) == {"k"}

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def test_intractable_5():
    factor_ik = random_gaussian(OrderedDict(i=Bint[2], k=Bint[2], x=Real))
    factor_j = random_gaussian(OrderedDict(j=Bint[2], y=Real))
    factor_ij = random_gaussian(OrderedDict(i=Bint[2], j=Bint[2], x=Real, y=Real))
    actual = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_ik, factor_j, factor_ij],
        eliminate=frozenset("ijxy"),
        plates=frozenset("ijk"),
    )
    assert set(actual.inputs) == {"k"}

    # Manually unroll plate j.
    factor_j0 = factor_j(j=0, y="y0")
    factor_j1 = factor_j(j=1, y="y1")
    factor_ij0 = factor_ij(j=0, y="y0")
    factor_ij1 = factor_ij(j=1, y="y1")
    expected = sum_product(
        ops.logaddexp,
        ops.add,
        [factor_ik, factor_j0, factor_j1, factor_ij0, factor_ij1],
        eliminate=frozenset(["i", "x", "y0", "y1"]),
        plates=frozenset("ik"),
    )
    assert set(expected.inputs) == {"k"}

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def test_var_in_plate_error():
    factor_i = random_gaussian(OrderedDict(i=Bint[2], x=Real, z=Real))
    with pytest.raises(ValueError):
        sum_product(
            ops.logaddexp,
            ops.add,
            [factor_i],
            eliminate=frozenset("ix"),
            plates=frozenset("i"),
            pedantic=True,
        )


@pytest.mark.xfail(reason="unclear semantics; incorrect computation of var_to_ordinal?")
def test_var_in_plate_ok():
    zs = Variable("zs", Reals[2])
    factor_i = random_gaussian(OrderedDict(i=Bint[2], x=Real, z=Real))
    factor_i = factor_i(z=zs["i"])
    with reflect:
        actual = sum_product(
            ops.logaddexp,
            ops.add,
            [factor_i],
            eliminate=frozenset("ix"),
            plates=frozenset("i"),
        )
    assert set(actual.inputs) == {"zs"}
    assert actual.inputs["zs"] == factor_i.inputs["zs"]


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (frozenset(), frozenset({"time", "x_0", "x_prev", "x_curr"})),
        (frozenset({"time", "x_0", "x_prev", "x_curr"}), frozenset()),
    ],
)
@pytest.mark.parametrize("x_dim,time", [(3, 1), (1, 5), (3, 5)])
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_0(
    impl, sum_op, prod_op, vars1, vars2, x_dim, time
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict({"time": Bint[time], "x_prev": Bint[x_dim], "x_curr": Bint[x_dim]})
    )

    factors = [f1, f2, f3]
    plate_to_step = {"time": frozenset({("x_0", "x_prev", "x_curr")})}

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (frozenset(), frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_curr"})),
        (frozenset({"y_0", "y_curr"}), frozenset({"time", "x_0", "x_prev", "x_curr"})),
        (frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_curr"}), frozenset()),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,time", [(2, 3, 5), (1, 3, 5), (2, 1, 5), (2, 3, 1)]
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_1(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, time
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict({"time": Bint[time], "x_prev": Bint[x_dim], "x_curr": Bint[x_dim]})
    )

    f4 = random_tensor(OrderedDict({"x_0": Bint[x_dim], "y_0": Bint[y_dim]}))

    f5 = random_tensor(
        OrderedDict({"time": Bint[time], "x_curr": Bint[x_dim], "y_curr": Bint[y_dim]})
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {"time": frozenset({("x_0", "x_prev", "x_curr")})}

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_prev", "y_curr"}),
        ),
        (
            frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_prev", "y_curr"}),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,time", [(2, 3, 5), (1, 3, 5), (2, 1, 5), (2, 3, 1)]
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_2(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, time
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict({"time": Bint[time], "x_prev": Bint[x_dim], "x_curr": Bint[x_dim]})
    )

    f4 = random_tensor(OrderedDict({"y_0": Bint[y_dim]}))

    f5 = random_tensor(
        OrderedDict({"time": Bint[time], "y_prev": Bint[y_dim], "y_curr": Bint[y_dim]})
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("y_0", "y_prev", "y_curr")})
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_prev", "y_curr"}),
        ),
        (
            frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_prev", "y_curr"}),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,time", [(2, 3, 5), (1, 3, 5), (2, 1, 5), (2, 3, 1)]
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_3(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, time
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict({"time": Bint[time], "x_prev": Bint[x_dim], "x_curr": Bint[x_dim]})
    )

    f4 = random_tensor(OrderedDict({"x_0": Bint[x_dim], "y_0": Bint[y_dim]}))

    f5 = random_tensor(
        OrderedDict(
            {
                "time": Bint[time],
                "x_curr": Bint[x_dim],
                "y_prev": Bint[y_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("y_0", "y_prev", "y_curr")})
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset(
                {"time", "x_0", "x_prev", "x_curr", "tones", "y_0", "y_prev", "y_curr"}
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,sequences,time,tones",
    [(2, 3, 2, 5, 4), (1, 3, 2, 5, 4), (2, 1, 2, 5, 4), (2, 3, 2, 1, 4)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_4(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, sequences, time, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {"sequences": Bint[sequences], "tones": Bint[tones], "y_0": Bint[y_dim]}
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "y_prev": Bint[y_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("y_0", "y_prev", "y_curr")}),
        "tones": {},
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "days",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "weeks",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset({"weeks", "y_0", "y_prev", "y_curr"}),
            frozenset({"sequences", "days", "tones", "x_0", "x_prev", "x_curr"}),
        ),
        (
            frozenset({"days", "tones", "x_0", "x_prev", "x_curr"}),
            frozenset({"sequences", "weeks", "y_0", "y_prev", "y_curr"}),
        ),
        (
            frozenset(
                {
                    "days",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "weeks",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "days",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "weeks",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,sequences,days,weeks,tones",
    [(2, 3, 2, 5, 4, 3), (1, 3, 2, 5, 4, 3), (2, 1, 2, 5, 4, 3), (2, 3, 2, 1, 4, 3)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_5(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, sequences, days, weeks, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(
        OrderedDict(
            {"sequences": Bint[sequences], "tones": Bint[tones], "x_0": Bint[x_dim]}
        )
    )

    f3 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "days": Bint[days],
                "tones": Bint[tones],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f4 = random_tensor(OrderedDict({"sequences": Bint[sequences], "y_0": Bint[y_dim]}))

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "weeks": Bint[weeks],
                "y_prev": Bint[y_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "sequences": {},
        "tones": {},
        "days": frozenset({("x_0", "x_prev", "x_curr")}),
        "weeks": frozenset({("y_0", "y_prev", "y_curr")}),
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset({"tones", "y_0", "y_curr"}),
            frozenset({"sequences", "time", "x_0", "x_prev", "x_curr"}),
        ),
        (
            frozenset({"time", "x_0", "x_prev", "x_curr", "tones", "y_0", "y_curr"}),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,sequences,time,tones",
    [(2, 3, 2, 5, 4), (1, 3, 2, 5, 4), (2, 1, 2, 5, 4), (2, 3, 2, 1, 4)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_6(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, sequences, time, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
            }
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr")}),
        "tones": {},
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset(
                {"time", "x_0", "x_prev", "x_curr", "tones", "y_0", "y_prev", "y_curr"}
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,sequences,time,tones",
    [(2, 3, 2, 5, 4), (1, 3, 2, 5, 4), (2, 1, 2, 5, 4), (2, 3, 2, 1, 4)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_7(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, sequences, time, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
            }
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "x_curr": Bint[x_dim],
                "y_prev": Bint[y_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("y_0", "y_prev", "y_curr")}),
        "tones": {},
    }

    with pytest.raises(ValueError, match="intractable!"):
        factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
        factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
        reduce(prod_op, factors2)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset({"tones", "y_0", "y_curr"}),
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                }
            ),
        ),
        (
            frozenset(
                {
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "w_dim,x_dim,y_dim,sequences,time,tones",
    [(3, 2, 3, 2, 5, 4), (3, 1, 3, 2, 5, 4), (3, 2, 1, 2, 5, 4), (3, 2, 3, 2, 1, 4)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_8(
    impl, sum_op, prod_op, vars1, vars2, w_dim, x_dim, y_dim, sequences, time, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "w_0": Bint[w_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "w_prev": Bint[w_dim],
                "w_curr": Bint[w_dim],
            }
        )
    )

    f4 = random_tensor(OrderedDict({"sequences": Bint[sequences], "x_0": Bint[x_dim]}))

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f6 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "w_0": Bint[w_dim],
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
            }
        )
    )

    f7 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "w_curr": Bint[w_dim],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5, f6, f7]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("w_0", "w_prev", "w_curr")}),
        "tones": {},
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("use_lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset({"tones", "y_0", "y_curr"}),
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                }
            ),
        ),
        (
            frozenset(
                {
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "w_dim,x_dim,y_dim,sequences,time,tones",
    [(3, 2, 3, 2, 5, 4), (3, 1, 3, 2, 5, 4), (3, 2, 1, 2, 5, 4), (3, 2, 3, 2, 1, 4)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_9(
    impl,
    use_lazy,
    sum_op,
    prod_op,
    vars1,
    vars2,
    w_dim,
    x_dim,
    y_dim,
    sequences,
    time,
    tones,
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "w_0": Bint[w_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "w_prev": Bint[w_dim],
                "w_curr": Bint[w_dim],
            }
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {"sequences": Bint[sequences], "w_0": Bint[w_dim], "x_0": Bint[x_dim]}
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "w_curr": Bint[w_dim],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f6 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "w_0": Bint[w_dim],
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
            }
        )
    )

    f7 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "w_curr": Bint[w_dim],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5, f6, f7]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("w_0", "w_prev", "w_curr")}),
        "tones": {},
    }

    with (lazy if use_lazy else eager):
        factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
        factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
        actual = reduce(prod_op, factors2)
    actual = apply_optimizer(actual)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset({"tones", "y_0", "y_curr"}),
            frozenset(
                {"sequences", "time", "w_0", "w_curr", "x_0", "x_prev", "x_curr"}
            ),
        ),
        (
            frozenset(
                {
                    "time",
                    "w_0",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "w_dim,x_dim,y_dim,sequences,time,tones",
    [(3, 2, 3, 2, 5, 4), (3, 1, 3, 2, 5, 4), (3, 2, 1, 2, 5, 4), (3, 2, 3, 2, 1, 4)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_10(
    impl, sum_op, prod_op, vars1, vars2, w_dim, x_dim, y_dim, sequences, time, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "w_0": Bint[w_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {"sequences": Bint[sequences], "time": Bint[time], "w_curr": Bint[w_dim]}
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {"sequences": Bint[sequences], "w_0": Bint[w_dim], "x_0": Bint[x_dim]}
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "w_curr": Bint[w_dim],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f6 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "w_0": Bint[w_dim],
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
            }
        )
    )

    f7 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "w_curr": Bint[w_dim],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5, f6, f7]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr")}),
        "tones": {},
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("use_lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "a",
                    "b",
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset({"tones", "y_0", "y_curr"}),
            frozenset(
                {
                    "a",
                    "b",
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                }
            ),
        ),
        (
            frozenset(
                {
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset({"a", "b", "sequences"}),
        ),
        (
            frozenset(
                {
                    "a",
                    "b",
                    "sequences",
                    "time",
                    "w_0",
                    "w_prev",
                    "w_curr",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y_0",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "a_dim,b_dim,w_dim,x_dim,y_dim,sequences,time,tones",
    [
        (2, 3, 3, 2, 3, 2, 3, 4),
        (2, 3, 3, 1, 3, 2, 5, 4),
        (2, 3, 3, 2, 1, 2, 5, 4),
        (2, 3, 3, 2, 3, 2, 1, 4),
    ],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_11(
    impl,
    use_lazy,
    sum_op,
    prod_op,
    vars1,
    vars2,
    a_dim,
    b_dim,
    w_dim,
    x_dim,
    y_dim,
    sequences,
    time,
    tones,
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"a": Bint[a_dim]}))

    f3 = random_tensor(OrderedDict({"sequences": Bint[sequences], "b": Bint[b_dim]}))

    f4 = random_tensor(
        OrderedDict(
            {"a": Bint[a_dim], "sequences": Bint[sequences], "w_0": Bint[w_dim]}
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "a": Bint[a_dim],
                "sequences": Bint[sequences],
                "time": Bint[time],
                "w_prev": Bint[w_dim],
                "w_curr": Bint[w_dim],
            }
        )
    )

    f6 = random_tensor(
        OrderedDict(
            {
                "b": Bint[b_dim],
                "sequences": Bint[sequences],
                "w_0": Bint[w_dim],
                "x_0": Bint[x_dim],
            }
        )
    )

    f7 = random_tensor(
        OrderedDict(
            {
                "b": Bint[b_dim],
                "sequences": Bint[sequences],
                "time": Bint[time],
                "w_curr": Bint[w_dim],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f8 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "w_0": Bint[w_dim],
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
            }
        )
    )

    f9 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "w_curr": Bint[w_dim],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("w_0", "w_prev", "w_curr")}),
        "tones": {},
    }

    with (lazy if use_lazy else eager):
        factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
        factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
        actual = reduce(prod_op, factors2)
    actual = apply_optimizer(actual)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_curr",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset(
                {
                    "time",
                    "w_0",
                    "w_curr",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "w_0",
                    "w_curr",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "w_dim,x_dim,y_dim,sequences,time,tones",
    [(3, 2, 3, 2, 5, 4), (3, 1, 3, 2, 5, 4), (3, 2, 1, 2, 5, 4), (3, 2, 3, 2, 1, 4)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_12(
    impl, sum_op, prod_op, vars1, vars2, w_dim, x_dim, y_dim, sequences, time, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "w_0": Bint[w_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {"sequences": Bint[sequences], "time": Bint[time], "w_curr": Bint[w_dim]}
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "w_0": Bint[w_dim],
                "x_0": Bint[x_dim],
            }
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "w_curr": Bint[w_dim],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f6 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "w_0": Bint[w_dim],
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
            }
        )
    )

    f7 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "w_curr": Bint[w_dim],
                "x_curr": Bint[x_dim],
                "y_prev": Bint[y_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5, f6, f7]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("y_0", "y_prev", "y_curr")}),
        "tones": {},
    }

    with pytest.raises(ValueError, match="intractable!"):
        factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
        factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
        reduce(prod_op, factors2)


@pytest.mark.parametrize("use_lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "w",
                    "days",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "weeks",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
        ),
        (
            frozenset({"weeks", "y_0", "y_prev", "y_curr"}),
            frozenset({"sequences", "w", "days", "tones", "x_0", "x_prev", "x_curr"}),
        ),
        (
            frozenset({"days", "tones", "x_0", "x_prev", "x_curr"}),
            frozenset({"sequences", "w", "weeks", "y_0", "y_prev", "y_curr"}),
        ),
        (
            frozenset(
                {
                    "days",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "weeks",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset({"sequences", "w"}),
        ),
        (
            frozenset(
                {
                    "w",
                    "days",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "weeks",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "w",
                    "days",
                    "tones",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "weeks",
                    "y_0",
                    "y_prev",
                    "y_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "w_dim,x_dim, y_dim, sequences, days, weeks, tones",
    [
        (3, 2, 3, 2, 5, 4, 3),
        (3, 1, 3, 2, 5, 4, 3),
        (3, 2, 1, 2, 5, 4, 3),
        (3, 2, 3, 2, 1, 4, 3),
    ],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_13(
    impl,
    use_lazy,
    sum_op,
    prod_op,
    vars1,
    vars2,
    w_dim,
    x_dim,
    y_dim,
    sequences,
    days,
    weeks,
    tones,
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(
        OrderedDict(
            {
                "w": Bint[w_dim],
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "x_0": Bint[x_dim],
            }
        )
    )

    f3 = random_tensor(
        OrderedDict(
            {
                "w": Bint[w_dim],
                "sequences": Bint[sequences],
                "days": Bint[days],
                "tones": Bint[tones],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {"w": Bint[w_dim], "sequences": Bint[sequences], "y_0": Bint[y_dim]}
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "w": Bint[w_dim],
                "sequences": Bint[sequences],
                "weeks": Bint[weeks],
                "y_prev": Bint[y_dim],
                "y_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "sequences": {},
        "tones": {},
        "days": frozenset({("x_0", "x_prev", "x_curr")}),
        "weeks": frozenset({("y_0", "y_prev", "y_curr")}),
    }

    with (lazy if use_lazy else eager):
        factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
        factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
        actual = reduce(prod_op, factors2)
    actual = apply_optimizer(actual)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y0_0",
                    "y0_prev",
                    "y0_curr",
                    "ycurr_0",
                    "ycurr_prev",
                    "ycurr_curr",
                }
            ),
        ),
        (
            frozenset(
                {
                    "tones",
                    "y0_0",
                    "y0_prev",
                    "y0_curr",
                    "ycurr_0",
                    "ycurr_prev",
                    "ycurr_curr",
                }
            ),
            frozenset({"sequences", "time", "x_0", "x_prev", "x_curr"}),
        ),
        (
            frozenset(
                {
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y0_0",
                    "y0_prev",
                    "y0_curr",
                    "ycurr_0",
                    "ycurr_prev",
                    "ycurr_curr",
                }
            ),
            frozenset({"sequences"}),
        ),
        (
            frozenset(
                {
                    "sequences",
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "tones",
                    "y0_0",
                    "y0_prev",
                    "y0_curr",
                    "ycurr_0",
                    "ycurr_prev",
                    "ycurr_curr",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,sequences,time,tones",
    [(2, 3, 2, 3, 2), (1, 3, 2, 3, 2), (2, 1, 2, 3, 2), (2, 3, 2, 1, 2)],
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_14(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, sequences, time, tones
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"sequences": Bint[sequences], "x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "x_prev": Bint[x_dim],
                "x_curr": Bint[x_dim],
            }
        )
    )

    f4 = random_tensor(
        OrderedDict(
            {"sequences": Bint[sequences], "x_0": Bint[x_dim], "y0_0": Bint[y_dim]}
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "tones": Bint[tones],
                "x_0": Bint[x_dim],
                "y0_prev": Bint[y_dim],
                "y0_curr": Bint[y_dim],
            }
        )
    )

    f6 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "x_curr": Bint[x_dim],
                "ycurr_0": Bint[y_dim],
            }
        )
    )

    f7 = random_tensor(
        OrderedDict(
            {
                "sequences": Bint[sequences],
                "time": Bint[time],
                "tones": Bint[tones],
                "x_curr": Bint[x_dim],
                "ycurr_prev": Bint[y_dim],
                "ycurr_curr": Bint[y_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5, f6, f7]
    plate_to_step = {
        "sequences": {},
        "time": frozenset({("x_0", "x_prev", "x_curr")}),
        "tones": frozenset(
            {("y0_0", "y0_prev", "y0_curr"), ("ycurr_0", "ycurr_prev", "ycurr_curr")}
        ),
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_prev", "y_curr"}),
        ),
        (
            frozenset({"time", "x_0", "x_prev", "x_curr", "y_0", "y_prev", "y_curr"}),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,time", [(2, 3, 5), (1, 3, 5), (2, 1, 5), (2, 3, 1)]
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_16(
    impl, sum_op, prod_op, vars1, vars2, x_dim, y_dim, time
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict({"time": Bint[time], "y_prev": Bint[y_dim], "x_curr": Bint[x_dim]})
    )

    f4 = random_tensor(OrderedDict({"y_0": Bint[y_dim]}))

    f5 = random_tensor(
        OrderedDict({"time": Bint[time], "x_prev": Bint[x_dim], "y_curr": Bint[y_dim]})
    )

    factors = [f1, f2, f3, f4, f5]
    plate_to_step = {
        "time": frozenset({("x_0", "x_prev", "x_curr"), ("y_0", "y_prev", "y_curr")})
    }

    factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
    factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
    actual = reduce(prod_op, factors2)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("use_lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "vars1,vars2",
    [
        (
            frozenset(),
            frozenset(
                {
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "y_0",
                    "y_curr",
                    "z0_0",
                    "z0",
                    "z1_0",
                    "z1",
                    "z2_0",
                    "z2",
                }
            ),
        ),
        (
            frozenset({"y_0", "y_curr", "z0_0", "z0", "z1_0", "z1", "z2_0", "z2"}),
            frozenset({"time", "x_0", "x_prev", "x_curr"}),
        ),
        (
            frozenset(
                {
                    "time",
                    "x_0",
                    "x_prev",
                    "x_curr",
                    "y_0",
                    "y_curr",
                    "z0_0",
                    "z0",
                    "z1_0",
                    "z1",
                    "z2_0",
                    "z2",
                }
            ),
            frozenset(),
        ),
    ],
)
@pytest.mark.parametrize(
    "x_dim,y_dim,z_dim,time", [(2, 3, 2, 5), (1, 3, 2, 5), (2, 1, 2, 5), (2, 3, 2, 1)]
)
@pytest.mark.parametrize(
    "sum_op,prod_op", [(ops.logaddexp, ops.add), (ops.add, ops.mul)]
)
@pytest.mark.parametrize(
    "impl", [modified_partial_sum_product, dynamic_partial_sum_product]
)
def test_modified_partial_sum_product_17(
    impl, use_lazy, sum_op, prod_op, vars1, vars2, x_dim, y_dim, z_dim, time
):

    f1 = random_tensor(OrderedDict({}))

    f2 = random_tensor(OrderedDict({"x_0": Bint[x_dim]}))

    f3 = random_tensor(
        OrderedDict({"time": Bint[time], "x_prev": Bint[x_dim], "x_curr": Bint[x_dim]})
    )

    f4 = random_tensor(
        OrderedDict(
            {
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
                "z0_0": Bint[z_dim],
                "z1_0": Bint[z_dim],
            }
        )
    )

    f5 = random_tensor(
        OrderedDict(
            {
                "time": Bint[time],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
                "z0": Bint[z_dim],
                "z1": Bint[z_dim],
            }
        )
    )

    f6 = random_tensor(
        OrderedDict(
            {
                "x_0": Bint[x_dim],
                "y_0": Bint[y_dim],
                "z1_0": Bint[z_dim],
                "z2_0": Bint[z_dim],
            }
        )
    )

    f7 = random_tensor(
        OrderedDict(
            {
                "time": Bint[time],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
                "z1": Bint[z_dim],
                "z2": Bint[z_dim],
            }
        )
    )

    f8 = random_tensor(
        OrderedDict({"x_0": Bint[x_dim], "y_0": Bint[y_dim], "z2_0": Bint[z_dim]})
    )

    f9 = random_tensor(
        OrderedDict(
            {
                "time": Bint[time],
                "x_curr": Bint[x_dim],
                "y_curr": Bint[y_dim],
                "z2": Bint[z_dim],
            }
        )
    )

    factors = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    plate_to_step = {"time": frozenset({("x_0", "x_prev", "x_curr")})}

    with (lazy if use_lazy else eager):
        factors1 = impl(sum_op, prod_op, factors, vars1, plate_to_step)
        factors2 = impl(sum_op, prod_op, factors1, vars2, plate_to_step)
        actual = reduce(prod_op, factors2)
    actual = apply_optimizer(actual)

    with lazy:
        unrolled_factors, unrolled_vars, remaining_plates = partial_unroll(
            factors, vars1 | vars2, plate_to_step
        )
        expected = reduce(prod_op, unrolled_factors).reduce(sum_op, unrolled_vars)
    expected = apply_optimizer(expected)

    assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("num_steps", [None] + list(range(1, 13)))
@pytest.mark.parametrize(
    "sum_op,prod_op,state_domain",
    [
        (ops.add, ops.mul, Bint[2]),
        (ops.add, ops.mul, Bint[3]),
        (ops.logaddexp, ops.add, Bint[2]),
        (ops.logaddexp, ops.add, Bint[3]),
        (ops.logaddexp, ops.add, Real),
        (ops.logaddexp, ops.add, Reals[2]),
    ],
    ids=str,
)
@pytest.mark.parametrize(
    "batch_inputs",
    [
        OrderedDict(),
        OrderedDict([("foo", Bint[5])]),
        OrderedDict([("foo", Bint[2]), ("bar", Bint[4])]),
    ],
    ids=lambda d: ",".join(d.keys()),
)
@pytest.mark.parametrize(
    "impl",
    [
        sequential_sum_product,
        naive_sequential_sum_product,
        MarkovProduct,
        partial(mixed_sequential_sum_product, num_segments=2),
        partial(mixed_sequential_sum_product, num_segments=3),
    ],
)
def test_sequential_sum_product(
    impl, sum_op, prod_op, batch_inputs, state_domain, num_steps
):
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
    operands = tuple(
        trans(time=t, prev="t_{}".format(t), curr="t_{}".format(t + 1))
        for t in range(num_steps)
    )
    reduce_vars = frozenset("t_{}".format(t) for t in range(1, num_steps))
    with reflect:
        expected = sum_product(sum_op, prod_op, operands, reduce_vars)
    expected = apply_optimizer(expected)
    expected = expected(**{"t_0": "prev", "t_{}".format(num_steps): "curr"})
    expected = expected.align(tuple(actual.inputs.keys()))
    with compress_gaussians:  # since actual,expected may differ only in rank
        actual = reinterpret(actual)
        expected = reinterpret(expected)
    assert_close(actual, expected, atol=1e-3 * num_steps, rtol=1e-3 * num_steps)


@pytest.mark.parametrize("num_steps", [None] + list(range(1, 6)))
@pytest.mark.parametrize(
    "batch_inputs",
    [
        OrderedDict(),
        OrderedDict([("foo", Bint[5])]),
        OrderedDict([("foo", Bint[2]), ("bar", Bint[4])]),
    ],
    ids=lambda d: ",".join(d.keys()),
)
@pytest.mark.parametrize(
    "x_domain,y_domain",
    [(Bint[2], Bint[3]), (Real, Reals[2, 2]), (Bint[2], Reals[2])],
    ids=str,
)
@pytest.mark.parametrize(
    "impl",
    [
        sequential_sum_product,
        naive_sequential_sum_product,
        MarkovProduct,
        partial(mixed_sequential_sum_product, num_segments=2),
        partial(mixed_sequential_sum_product, num_segments=3),
    ],
)
def test_sequential_sum_product_multi(
    impl, x_domain, y_domain, batch_inputs, num_steps
):
    sum_op = ops.logaddexp
    prod_op = ops.add
    inputs = OrderedDict(batch_inputs)
    inputs.update(x_prev=x_domain, x_curr=x_domain, y_prev=y_domain, y_curr=y_domain)
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

    with moment_matching:
        actual = impl(sum_op, prod_op, trans, time, step)
        expected_inputs = batch_inputs.copy()
        expected_inputs.update(
            x_prev=x_domain, x_curr=x_domain, y_prev=y_domain, y_curr=y_domain
        )
        assert dict(actual.inputs) == expected_inputs

        # Check against contract.
        operands = tuple(
            trans(
                time=t,
                x_prev="x_{}".format(t),
                x_curr="x_{}".format(t + 1),
                y_prev="y_{}".format(t),
                y_curr="y_{}".format(t + 1),
            )
            for t in range(num_steps)
        )
        reduce_vars = frozenset("x_{}".format(t) for t in range(1, num_steps)).union(
            "y_{}".format(t) for t in range(1, num_steps)
        )
        expected = sum_product(sum_op, prod_op, operands, reduce_vars)
        expected = expected(
            **{
                "x_0": "x_prev",
                "x_{}".format(num_steps): "x_curr",
                "y_0": "y_prev",
                "y_{}".format(num_steps): "y_curr",
            }
        )
        expected = expected.align(tuple(actual.inputs.keys()))


@pytest.mark.parametrize("num_steps", [1, 2, 3, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sequential_sum_product_bias_1(num_steps, dim):
    time = Variable("time", Bint[num_steps])
    bias_dist = random_gaussian(OrderedDict([("bias", Reals[dim])]))
    trans = random_gaussian(
        OrderedDict(
            [("time", Bint[num_steps]), ("x_prev", Reals[dim]), ("x_curr", Reals[dim])]
        )
    )
    obs = random_gaussian(
        OrderedDict(
            [("time", Bint[num_steps]), ("x_curr", Reals[dim]), ("bias", Reals[dim])]
        )
    )
    factor = trans + obs + bias_dist
    assert set(factor.inputs) == {"time", "bias", "x_prev", "x_curr"}

    result = sequential_sum_product(
        ops.logaddexp, ops.add, factor, time, {"x_prev": "x_curr"}
    )
    assert set(result.inputs) == {"bias", "x_prev", "x_curr"}


@pytest.mark.xfail(reason="missing pattern for Gaussian(x=y[z]) for real x")
@pytest.mark.parametrize("num_steps", [1, 2, 3, 10])
@pytest.mark.parametrize("num_sensors", [2])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sequential_sum_product_bias_2(num_steps, num_sensors, dim):
    time = Variable("time", Bint[num_steps])
    bias = Variable("bias", Reals[num_sensors, dim])
    bias_dist = random_gaussian(OrderedDict([("bias", Reals[num_sensors, dim])]))
    trans = random_gaussian(
        OrderedDict(
            [("time", Bint[num_steps]), ("x_prev", Reals[dim]), ("x_curr", Reals[dim])]
        )
    )
    obs = random_gaussian(
        OrderedDict(
            [("time", Bint[num_steps]), ("x_curr", Reals[dim]), ("bias", Reals[dim])]
        )
    )

    # Each time step only a single sensor observes x,
    # and each sensor has a different bias.
    sensor_id = Tensor(
        ops.new_arange(get_default_prototype(), num_steps) % 2,
        OrderedDict(time=Bint[num_steps]),
        dtype=2,
    )
    with eager_or_die:
        factor = trans + obs(bias=bias[sensor_id]) + bias_dist
    assert set(factor.inputs) == {"time", "bias", "x_prev", "x_curr"}

    result = sequential_sum_product(
        ops.logaddexp, ops.add, factor, time, {"x_prev": "x_curr"}
    )
    assert set(result.inputs) == {"bias", "x_prev", "x_curr"}


def _check_sarkka_bilmes(trans, expected_inputs, global_vars, num_periods=1):

    sum_op, prod_op = ops.logaddexp, ops.add

    assert "time" in trans.inputs
    duration = trans.inputs["time"].dtype
    time_var = Variable("time", Bint[duration])

    expected = naive_sarkka_bilmes_product(
        sum_op, prod_op, trans, time_var, global_vars
    )
    assert dict(expected.inputs) == expected_inputs

    actual = sarkka_bilmes_product(
        sum_op, prod_op, trans, time_var, global_vars, num_periods=num_periods
    )
    assert dict(actual.inputs) == expected_inputs

    actual = actual.align(tuple(expected.inputs.keys()))
    with compress_gaussians:
        actual = reinterpret(actual)
        expected = reinterpret(expected)
    assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6])
def test_sarkka_bilmes_example_0(duration):

    trans = random_tensor(OrderedDict({"time": Bint[duration], "a": Bint[3]}))

    expected_inputs = {"a": Bint[3]}

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6])
def test_sarkka_bilmes_example_1(duration):

    trans = random_tensor(
        OrderedDict(
            {"time": Bint[duration], "a": Bint[3], "b": Bint[2], "_PREV_b": Bint[2]}
        )
    )

    expected_inputs = {"a": Bint[3], "b": Bint[2], "_PREV_b": Bint[2]}

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6, 7, 8])
def test_sarkka_bilmes_example_2(duration):

    trans = random_tensor(
        OrderedDict(
            {
                "time": Bint[duration],
                "a": Bint[4],
                "b": Bint[3],
                "_PREV_b": Bint[3],
                "c": Bint[2],
                "_PREV__PREV_c": Bint[2],
            }
        )
    )

    expected_inputs = {
        "a": Bint[4],
        "b": Bint[3],
        "_PREV_b": Bint[3],
        "c": Bint[2],
        "_PREV__PREV_c": Bint[2],
        "_PREV_c": Bint[2],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6, 7, 8])
def test_sarkka_bilmes_example_3(duration):

    trans = random_tensor(
        OrderedDict(
            {
                "time": Bint[duration],
                "a": Bint[4],
                "c": Bint[2],
                "_PREV__PREV_c": Bint[2],
            }
        )
    )

    expected_inputs = {
        "a": Bint[4],
        "c": Bint[2],
        "_PREV__PREV_c": Bint[2],
        "_PREV_c": Bint[2],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())


@pytest.mark.parametrize("duration", [3, 4, 5, 6, 7, 9])
def test_sarkka_bilmes_example_4(duration):

    trans = random_tensor(
        OrderedDict(
            {
                "time": Bint[duration],
                "a": Bint[2],
                "_PREV_a": Bint[2],
                "_PREV__PREV__PREV_a": Bint[2],
            }
        )
    )

    expected_inputs = {
        "a": Bint[2],
        "_PREV__PREV_a": Bint[2],
        "_PREV__PREV__PREV_a": Bint[2],
        "_PREV_a": Bint[2],
    }

    _check_sarkka_bilmes(trans, expected_inputs, frozenset())


@pytest.mark.parametrize("duration", [2, 3, 4, 5, 6])
def test_sarkka_bilmes_example_5(duration):

    trans = random_tensor(
        OrderedDict(
            {"time": Bint[duration], "a": Bint[3], "_PREV_a": Bint[3], "x": Bint[2]}
        )
    )

    expected_inputs = {"a": Bint[3], "_PREV_a": Bint[3], "x": Bint[2]}

    global_vars = frozenset(["x"])

    _check_sarkka_bilmes(trans, expected_inputs, global_vars)


@pytest.mark.parametrize("duration", [3, 4, 5, 6, 7, 8, 9])
def test_sarkka_bilmes_example_6(duration):

    trans = random_tensor(
        OrderedDict(
            {
                "time": Bint[duration],
                "a": Bint[2],
                "_PREV_a": Bint[2],
                "_PREV__PREV__PREV_a": Bint[2],
                "x": Bint[3],
            }
        )
    )

    expected_inputs = {
        "a": Bint[2],
        "_PREV__PREV_a": Bint[2],
        "_PREV__PREV__PREV_a": Bint[2],
        "_PREV_a": Bint[2],
        "x": Bint[3],
    }

    global_vars = frozenset(["x"])

    _check_sarkka_bilmes(trans, expected_inputs, global_vars)


@pytest.mark.parametrize("time_input", [("time", Bint[t]) for t in range(6, 11)])
@pytest.mark.parametrize("global_inputs", [(), (("x", Bint[2]),)])
@pytest.mark.parametrize(
    "local_inputs",
    [
        # tensor
        (("a", Bint[2]),),
        (("a", Bint[2]), ("_PREV_a", Bint[2])),
        (("a", Bint[2]), ("b", Bint[2]), ("_PREV_b", Bint[2])),
        (("a", Bint[2]), ("b", Bint[2]), ("_PREV__PREV_b", Bint[2])),
        (
            ("a", Bint[2]),
            ("b", Bint[2]),
            ("_PREV_b", Bint[2]),
            ("c", Bint[2]),
            ("_PREV__PREV_c", Bint[2]),
        ),
        (("a", Bint[2]), ("_PREV_a", Bint[2]), ("_PREV__PREV__PREV_a", Bint[2])),
        (
            ("a", Bint[2]),
            ("b", Bint[2]),
            ("_PREV__PREV_b", Bint[2]),
            ("_PREV__PREV__PREV_a", Bint[2]),
        ),
        # gaussian
        (("a", Real),),
        (("a", Real), ("_PREV_a", Real)),
        (("a", Real), ("b", Real), ("_PREV_b", Real)),
        (("a", Real), ("b", Real), ("_PREV__PREV_b", Real)),
        (
            ("a", Real),
            ("b", Real),
            ("_PREV_b", Real),
            ("c", Real),
            ("_PREV__PREV_c", Real),
        ),
        (("a", Real), ("_PREV_a", Real), ("_PREV__PREV__PREV_a", Real)),
        (
            ("a", Real),
            ("b", Real),
            ("_PREV__PREV_b", Real),
            ("_PREV__PREV__PREV_a", Real),
        ),
        # mv gaussian
        (("a", Reals[2]), ("b", Reals[2]), ("_PREV_b", Reals[2])),
        (("a", Reals[2]), ("b", Reals[2]), ("_PREV__PREV_b", Reals[2])),
    ],
)
@pytest.mark.parametrize("num_periods", [1, 2])
def test_sarkka_bilmes_generic(time_input, global_inputs, local_inputs, num_periods):

    lags = {
        kk: reduce(
            max,
            [
                len(re.search("^(_PREV_)*", k).group(0)) // 6
                for k, v in local_inputs
                if k.strip("_PREV_") == kk
            ],
            0,
        )
        for kk, vv in local_inputs
        if not kk.startswith("_PREV_")
    }
    expected_inputs = dict(
        global_inputs
        + tuple(
            set(
                ((t * "_PREV_" + k), v)
                for k, v in local_inputs
                if not k.startswith("_PREV_")
                for t in range(0, lags[k] + 1)
            )
        )
    )

    trans_inputs = OrderedDict(global_inputs + (time_input,) + local_inputs)
    global_vars = frozenset(k for k, v in global_inputs)

    if any(v.dtype == "real" for v in trans_inputs.values()):
        trans = random_gaussian(trans_inputs)
    else:
        trans = random_tensor(trans_inputs)

    _check_sarkka_bilmes(trans, expected_inputs, global_vars, num_periods)


@pytest.mark.parametrize(
    "duration,num_segments", [(12, 1), (12, 2), (12, 3), (12, 4), (12, 6)]
)
def test_mixed_sequential_sum_product(duration, num_segments):

    sum_op, prod_op = ops.logaddexp, ops.add
    time_var = Variable("time", Bint[duration])
    step = {"_PREV_x": "x"}

    trans_inputs = (
        ((time_var.name, Bint[duration]),)
        + tuple((k, Bint[2]) for k in step.keys())
        + tuple((v, Bint[2]) for v in step.values())
    )

    trans = random_tensor(OrderedDict(trans_inputs))

    expected = sequential_sum_product(sum_op, prod_op, trans, time_var, step)
    actual = mixed_sequential_sum_product(
        sum_op, prod_op, trans, time_var, step, num_segments=num_segments
    )

    assert_close(actual, expected)
