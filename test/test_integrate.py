from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import pytest
import torch

import funsor

import funsor.ops as ops
from funsor.domains import bint
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import Finitary, optimize
from funsor.terms import reflect
from funsor.testing import assert_close, make_einsum_example
from funsor.torch import Tensor

from funsor.einsum import einsum, naive_einsum
from funsor.integrate import Integrate, naive_integrate_einsum, naive_integrate


EINSUM_EXAMPLES = [
    "a,b->",
    "ab,a->",
    # "a,a->",
    # "a,a,a,ab->",
    # "ab->",
    # "ab,bc,cd->",
    # "abc,bcd,def->",
    # "abc,abc,bcd,bcd,def,def->",
    # "ab,bc,cd,de->",
    # "ab,ab,bc,bc,cd,cd->",
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend,fill', [
    ('torch', None),
    # ('torch', 1.),
    # ('pyro.ops.einsum.torch_log', None)
])
def test_integrate_einsum_product_measure(equation, backend, fill):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation, fill=fill)

    # extras = [Tensor(torch.ones((sizes[v],)) / float(sizes[v]), 
    #                  OrderedDict([(v, bint(sizes[v]))]))
    #           for inp in inputs for v in inp]
    extras = []
    with interpretation(reflect):
        expected = einsum(equation, *funsor_operands + extras, backend=backend)
        print("TRUE GRAPH: {}".format(expected))
    expected = reinterpret(expected)
    actual = naive_integrate_einsum(equation, *funsor_operands, backend=backend)

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    print(expected / actual, actual / expected)
    assert_close(expected, actual, atol=1e-4)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


@pytest.mark.xfail(reason="wtf?")
@pytest.mark.parametrize('equation1,equation2',
                         list(zip(EINSUM_EXAMPLES, EINSUM_EXAMPLES)))
def test_integrate_naive_pair(equation1, equation2):

    # identical structure
    funsor_operands1 = [a.abs() for a in make_einsum_example(equation1)[-1]]
    funsor_operands2 = [a.abs() for a in make_einsum_example(equation2)[-1]]

    with interpretation(reflect):
        measure = Finitary(ops.mul, tuple(funsor_operands1))
        integrand = Finitary(ops.mul, tuple(funsor_operands2))

    expected = naive_integrate(measure, integrand)
    with interpretation(optimize):
        actual = Integrate(measure, integrand)
    actual = reinterpret(actual)

    assert_close(expected, actual, atol=1e-4)
