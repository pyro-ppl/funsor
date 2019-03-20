from __future__ import absolute_import, division, print_function

from collections import OrderedDict  # noqa: F401
import pytest
import torch  # noqa: F401

import funsor

import funsor.ops as ops
from funsor.domains import bint  # noqa: F401
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import Finitary, optimize
from funsor.terms import reflect
from funsor.testing import assert_close, make_einsum_example
from funsor.torch import Tensor  # noqa: F401

from funsor.einsum import einsum
from funsor.contract import Contract, naive_contract_einsum


EINSUM_EXAMPLES = [
    "a,b->",
    "ab,a->",
    "a,a->",
    "a,a,a,ab->",
    "ab->",
    "ab,bc,cd->",
    "abc,bcd,def->",
    "abc,abc,bcd,bcd,def,def->",
    "ab,bc,cd,de->",
    "ab,ab,bc,bc,cd,cd->",
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend,fill', [
    ('torch', None),
    ('torch', 1.),
    ('pyro.ops.einsum.torch_log', None)
])
def test_contract_einsum_product_measure(equation, backend, fill):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation, fill=fill)

    with interpretation(reflect):
        expected = einsum(equation, *funsor_operands, backend=backend)
        print("TRUE GRAPH: {}".format(expected))
    expected = reinterpret(expected)
    actual = naive_contract_einsum(equation, *funsor_operands, backend=backend)

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    print(expected / actual, actual / expected)
    assert_close(expected, actual, atol=1e-4)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


@pytest.mark.parametrize('equation1', EINSUM_EXAMPLES)
@pytest.mark.parametrize('equation2', EINSUM_EXAMPLES)
def test_contract_naive_pair(equation1, equation2):

    # identical structure
    case1 = make_einsum_example(equation1)
    case2 = make_einsum_example(equation2)
    sizes1, funsor_operands1 = case1[2], case1[-1]
    sizes2, funsor_operands2 = case2[2], case2[-1]

    assert all(sizes1[k] == sizes2[k] for k in set(sizes1.keys()) & set(sizes2.keys()))

    with interpretation(optimize):
        measure = Finitary(ops.mul, tuple(funsor_operands1))
        integrand = Finitary(ops.mul, tuple(funsor_operands2))

        expected = (measure * integrand).reduce(ops.add)

        actual = Contract(measure, integrand,
                          frozenset(measure.inputs) | frozenset(integrand.inputs))

    actual = reinterpret(actual)
    expected = reinterpret(expected)

    assert_close(expected, actual, atol=1e-4, rtol=1e-4)
