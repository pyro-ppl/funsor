from __future__ import absolute_import, division, print_function

import pytest
import opt_einsum
import torch

import funsor

import funsor.ops as ops
from funsor.interpreter import interpretation
from funsor.optimizer import Finitary
from funsor.terms import reflect
from funsor.testing import assert_close, make_einsum_example

from funsor.einsum import naive_einsum
from funsor.integrate import integrate, naive_integrate_einsum, naive_integrate


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
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
def test_integrate_einsum_product_measure(equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = opt_einsum.contract(equation, *operands, backend=backend)

    actual = naive_einsum(equation, *funsor_operands, backend=backend)
    actual_integrate = naive_integrate_einsum(equation, *funsor_operands, backend=backend)

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))
        actual_integrate = actual_integrate.align(tuple(outputs[0]))

    print(actual / actual_integrate)
    assert_close(actual, actual_integrate, atol=1e-4)
    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


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
    actual = integrate(measure, integrand)

    assert_close(expected, actual, atol=1e-4)
