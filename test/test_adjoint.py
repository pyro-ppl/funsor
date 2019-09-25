from collections import OrderedDict

import opt_einsum
import pytest
import torch
from pyro.ops.contract import einsum as pyro_einsum
from pyro.ops.einsum.adjoint import require_backward as pyro_require_backward

import funsor
import funsor.ops as ops
from funsor.adjoint import AdjointTape
from funsor.domains import bint
from funsor.einsum import BACKEND_ADJOINT_OPS, einsum, naive_einsum, naive_plated_einsum
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer
from funsor.sum_product import (
    MarkovProduct,
    naive_sequential_sum_product,
    sequential_sum_product,
    sum_product
)
from funsor.terms import Slice, Subs, Variable, reflect
from funsor.testing import assert_close, make_einsum_example, make_plated_hmm_einsum, random_tensor, xfail_param


EINSUM_EXAMPLES = [
    "a->",
    "ab->",
    ",->",
    ",,->",
    "a,a->a",
    "a,a,a->a",
    "a,b->",
    "ab,a->",
    "a,b,c->",
    "a,a->",
    "a,a,a,ab->",
    "abc,bcd,cde->",
    "ab,bc,cd->",
    "ab,b,bc,c,cd,d->",
]


@pytest.mark.parametrize('einsum_impl', [naive_einsum, einsum])
@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    'pyro.ops.einsum.torch_marginal',
    xfail_param('pyro.ops.einsum.torch_map', reason="wrong adjoint"),
])
def test_einsum_adjoint(einsum_impl, equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    with AdjointTape() as tape:  # interpretation(reflect):
        fwd_expr = einsum_impl(equation, *funsor_operands, backend=backend)
    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(equation, *operands,
                               modulo_total=True,
                               backend=backend)[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        actual = actuals[fv]
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


@pytest.mark.parametrize('einsum_impl', [naive_einsum, einsum])
@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    'pyro.ops.einsum.torch_marginal',
    xfail_param('pyro.ops.einsum.torch_map', reason="wrong adjoint"),
])
def test_einsum_adjoint_unary_marginals(einsum_impl, equation, backend):
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    with AdjointTape() as tape:  # interpretation(reflect):
        inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
        equation = ",".join(inputs) + "->"
        targets = [Variable(k, bint(sizes[k])) for k in set(sizes)]
        fwd_expr = einsum_impl(equation, *funsor_operands, backend=backend)
    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, targets)

    for target in targets:
        actual = actuals[target]

        expected = opt_einsum.contract(equation + target.name, *operands,
                                       backend=backend)
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


PLATED_EINSUM_EXAMPLES = [
    (',i->', 'i'),
    ('i->', 'i'),
    ('ai->', 'i'),
    (',ai,abij->', 'ij'),
    ('a,ai,bij->', 'ij'),
    ('ai,abi,bci,cdi->', 'i'),
    ('aij,abij,bcij->', 'ij'),
    ('a,abi,bcij,cdij->', 'ij'),
]


@pytest.mark.parametrize('einsum_impl', [naive_plated_einsum, einsum])
@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    'pyro.ops.einsum.torch_marginal',
    xfail_param('pyro.ops.einsum.torch_map', reason="wrong adjoint"),
])
def test_plated_einsum_adjoint(einsum_impl, equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    with AdjointTape() as tape:  # interpretation(reflect):
        fwd_expr = einsum_impl(equation, *funsor_operands, plates=plates, backend=backend)
    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(equation, *operands,
                               modulo_total=False,
                               plates=plates,
                               backend=backend)[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        actual = actuals[fv]
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


OPTIMIZED_PLATED_EINSUM_EXAMPLES = [
    make_plated_hmm_einsum(num_steps, num_obs_plates=b, num_hidden_plates=a)
    for num_steps in range(20, 50, 6)
    for (a, b) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
]


@pytest.mark.parametrize('equation,plates', OPTIMIZED_PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    'pyro.ops.einsum.torch_marginal',
    xfail_param('pyro.ops.einsum.torch_map', reason="wrong adjoint"),
])
def test_optimized_plated_einsum_adjoint(equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    with AdjointTape() as tape:  # interpretation(reflect):
        fwd_expr = einsum(equation, *funsor_operands, plates=plates, backend=backend)
    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(equation, *operands,
                               modulo_total=False,
                               plates=plates,
                               backend=backend)[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        actual = actuals[fv]
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


def test_slice_adjoint_simple():

    inputs = OrderedDict([('i', bint(7)), ('j', bint(5)), ('k', bint(3))])
    x = random_tensor(inputs)
    x.data.requires_grad = True
    s = Slice('i', 0, 7, 2, 7)

    with AdjointTape() as tape:
        fwd_expr = Subs(x, (('i', s),)).reduce(ops.add)

    actual = tape.adjoint(ops.add, ops.mul, fwd_expr, (x,))[x]

    expected = torch.autograd.grad(fwd_expr.data.sum(), (x.data,))[0]
    assert actual.data.shape == expected.data.shape
    assert_close(actual.data, expected * x.data)


@pytest.mark.parametrize('num_steps', list(range(3, 13)))
@pytest.mark.parametrize('sum_op,prod_op,state_domain', [
    (ops.add, ops.mul, bint(2)),
    (ops.add, ops.mul, bint(3)),
    # TODO define unit for ops.logaddexp
    # (ops.logaddexp, ops.add, bint(2)),
    # (ops.logaddexp, ops.add, bint(3)),
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
def test_sequential_sum_product_adjoint_discrete(impl, sum_op, prod_op, batch_inputs, state_domain, num_steps):
    inputs = OrderedDict(batch_inputs)
    inputs.update(prev=state_domain, curr=state_domain)
    inputs["time"] = bint(num_steps)
    trans = random_tensor(inputs)
    time = Variable("time", bint(num_steps))

    with AdjointTape() as actual_tape:
        actual = impl(sum_op, prod_op, trans, time, {"prev": "curr"})
    actual_bwd = actual_tape.adjoint(sum_op, prod_op, actual, (trans,))[trans]

    expected_inputs = batch_inputs.copy()
    expected_inputs.update(prev=state_domain, curr=state_domain)
    assert dict(actual.inputs) == expected_inputs

    # Check against contract.
    operands = tuple(trans(time=t, prev="t_{}".format(t), curr="t_{}".format(t+1))
                     for t in range(num_steps))
    reduce_vars = frozenset("t_{}".format(t) for t in range(1, num_steps))
    with AdjointTape() as expected_tape:
        with interpretation(reflect):
            expected = sum_product(sum_op, prod_op, operands, reduce_vars)
        expected = apply_optimizer(expected)
        expected = expected(**{"t_0": "prev", "t_{}".format(num_steps): "curr"})
        expected = expected.align(tuple(actual.inputs.keys()))

    expected_bwds = expected_tape.adjoint(sum_op, prod_op, expected, operands)

    # check forward pass
    assert_close(actual, expected, rtol=5e-4 * num_steps)

    # check backward pass
    for t, operand in enumerate(operands):
        expected_bwd = expected_bwds[operand]
        assert_close(actual_bwd(t=t), expected_bwd)
