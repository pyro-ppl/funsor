from __future__ import absolute_import, division, print_function

import opt_einsum
import pytest
import torch

import funsor
import funsor.ops as ops
from funsor.contract import contract
from funsor.testing import assert_close

EQUATIONS = [
    'ab,bc->abc',
    'ab,bc->ac',
    'ab,bc->',
]


@pytest.mark.parametrize('equation', EQUATIONS)
def test_sumproduct(equation):
    inputs, output = equation.split('->')
    dims = tuple(output)
    lhs_dims, rhs_dims = inputs.split(',')
    lhs_dims = tuple(lhs_dims)
    rhs_dims = tuple(rhs_dims)
    sizes = {'a': 2, 'b': 3, 'c': 4}
    lhs = funsor.Tensor(lhs_dims, 0.5 + torch.rand(*(sizes[d] for d in lhs_dims)))
    rhs = funsor.Tensor(rhs_dims, 0.5 + torch.rand(*(sizes[d] for d in rhs_dims)))
    reduce_dims = frozenset(lhs_dims + rhs_dims) - frozenset(output)

    expected_data = opt_einsum.contract(equation, lhs.data, rhs.data,
                                        backend='torch')
    expected = funsor.Tensor(dims, expected_data)
    actual = contract((ops.add, ops.mul), lhs, rhs, reduce_dims)
    assert_close(actual, expected)


@pytest.mark.parametrize('equation', EQUATIONS)
def test_logsumproductexp(equation):
    inputs, output = equation.split('->')
    dims = tuple(output)
    lhs_dims, rhs_dims = inputs.split(',')
    lhs_dims = tuple(lhs_dims)
    rhs_dims = tuple(rhs_dims)
    sizes = {'a': 2, 'b': 3, 'c': 4}
    lhs = funsor.Tensor(lhs_dims, 0.5 + torch.rand(*(sizes[d] for d in lhs_dims)))
    rhs = funsor.Tensor(rhs_dims, 0.5 + torch.rand(*(sizes[d] for d in rhs_dims)))
    reduce_dims = frozenset(lhs_dims + rhs_dims) - frozenset(output)

    expected_data = opt_einsum.contract(equation, lhs.data, rhs.data,
                                        backend='pyro.ops.einsum.torch_log')
    expected = funsor.Tensor(dims, expected_data)
    actual = contract((ops.logaddexp, ops.add), lhs, rhs, reduce_dims)
    assert_close(actual, expected)
