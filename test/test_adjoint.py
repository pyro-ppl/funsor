from collections import OrderedDict
from functools import reduce

import pytest
import torch
from pyro.ops.contract import einsum as pyro_einsum
from pyro.ops.einsum.adjoint import require_backward as pyro_require_backward

import funsor
import funsor.ops as ops
from funsor.adjoint import AdjointTape
from funsor.domains import bint, reals
from funsor.einsum import BACKEND_ADJOINT_OPS, einsum, naive_plated_einsum
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer
from funsor.sum_product import (
    MarkovProduct,
    naive_sequential_sum_product,
    sequential_sum_product,
    sum_product
)
from funsor.terms import Binary, Variable, lazy, reflect, to_funsor
from funsor.testing import (
    assert_close,
    check_funsor,
    make_einsum_example,
    make_plated_hmm_einsum,
    random_tensor,
    xfail_param,
)


EINSUM_EXAMPLES = [
    "a,b,c->abc",
    "ab,bc->abc",
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


PLATED_EINSUM_EXAMPLES = [
    ('i->', 'i'),
    (',i->', 'i'),
    ('ai->', 'i'),
    (',ai,abij->', 'ij'),
    ('a,ai,bij->', 'ij'),
    ('ai,abi,bci,cdi->', 'i'),
    ('aij,abij,bcij->', 'ij'),
    ('a,abi,bcij,cdij->', 'ij'),
]


@pytest.mark.parametrize('einsum_impl', [naive_plated_einsum, einsum])
@pytest.mark.parametrize('equation,plates', [(e, "") for e in EINSUM_EXAMPLES] + PLATED_EINSUM_EXAMPLES)
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
                               modulo_total=True,
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


@pytest.mark.parametrize('lhs_equation,lhs_plates', [(e, "") for e in EINSUM_EXAMPLES[:5]] + PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('rhs_equation,rhs_plates', [(e, "") for e in EINSUM_EXAMPLES[:5]] + PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    'pyro.ops.einsum.torch_marginal',
    xfail_param('pyro.ops.einsum.torch_map', reason="wrong adjoint"),
])
def test_adjoint_binary_sum(lhs_equation, lhs_plates, rhs_equation, rhs_plates, backend):
    lhs_inputs, lhs_outputs, lhs_sizes, lhs_operands, lhs_funsor_operands = make_einsum_example(lhs_equation)
    rhs_inputs, rhs_outputs, rhs_sizes, rhs_operands, rhs_funsor_operands = make_einsum_example(rhs_equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    # make up numbers
    const1 = to_funsor(torch.tensor(0.5231))
    const2 = to_funsor(torch.tensor(0.8376))

    with AdjointTape() as tape:  # interpretation(reflect):
        lhs = einsum(lhs_equation, *lhs_funsor_operands, plates=lhs_plates, backend=backend)
        rhs = einsum(rhs_equation, *rhs_funsor_operands, plates=rhs_plates, backend=backend)
        fwd_expr = Binary(sum_op, Binary(prod_op, const1, lhs), Binary(prod_op, const2, rhs))

    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, lhs_funsor_operands + rhs_funsor_operands)

    for operand in lhs_operands:
        pyro_require_backward(operand)
    expected_lhs_out = pyro_einsum(lhs_equation, *lhs_operands,
                                   plates=lhs_plates, modulo_total=True, backend=backend)[0]
    expected_lhs_out._pyro_backward()

    for operand in rhs_operands:
        pyro_require_backward(operand)
    expected_rhs_out = pyro_einsum(rhs_equation, *rhs_operands,
                                   plates=rhs_plates, modulo_total=True, backend=backend)[0]
    expected_rhs_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(lhs_inputs + rhs_inputs,
                                          lhs_operands + rhs_operands,
                                          lhs_funsor_operands + rhs_funsor_operands)):
        actual = actuals[fv]
        const = const1 if any(tv is o for o in lhs_operands) else const2
        expected = prod_op(tv._pyro_backward_result, const.data)
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


@pytest.mark.parametrize("equation,plates", [(e, "") for e in EINSUM_EXAMPLES[:5]] + PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    'pyro.ops.einsum.torch_marginal',
])
def test_adjoint_involution(equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    # forward-backward once
    with AdjointTape() as tape:
        fwd_expr = einsum(equation, *funsor_operands, plates=plates, backend=backend)
    with AdjointTape() as tape2:
        bwd_exprs = tape.adjoint(sum_op, prod_op, fwd_expr, funsor_operands)
        bwd_expr = reduce(lambda a, b: Binary(sum_op, a, b), bwd_exprs.values())

    # backward again
    actual = tape2.adjoint(sum_op, prod_op, bwd_expr, (fwd_expr,))

    assert_close(actual[fwd_expr], fwd_expr)


ADJOINT_BINARY_TESTS = [
    ('a * x', {'x': 'a'}),
    ('b * x', {'x': 'b'}),
    ('c * x', {'x': 'c'}),
    ('a * x + b', {'x': 'a'}),
    ('x + y', {'x': '1.', 'y': '1.'}),
    ('x * y', {'x': '1. * y', 'y': '1. * x'}),
    ('a * x + b * (y + z + c) + d', {'x': 'a', 'y': 'b', 'z': 'b'}),
]


@pytest.mark.parametrize('expr,expected_adjoint_exprs', ADJOINT_BINARY_TESTS)
def test_adjoint_binary_only(expr, expected_adjoint_exprs):

    x, y, z = [Variable(n, reals()) for n in ('x', 'y', 'z')]
    a, b, c, d = [random_tensor(OrderedDict(inputs))
                  for inputs in ({'i': bint(2)}, {'j': bint(3)}, {'i': bint(2), 'j': bint(3)}, {})]
    locals_ = locals()

    expected_adjoints = OrderedDict((eval(k, locals_), to_funsor(eval(v, locals_)))
                                    for k, v in expected_adjoint_exprs.items())

    with interpretation(lazy), AdjointTape() as tape:
        fwd_expr = eval(expr, locals_)

    actual_adjoints = tape.adjoint(ops.add, ops.mul, fwd_expr, list(expected_adjoints.keys()))

    assert list(actual_adjoints.keys()) == list(expected_adjoints.keys())
    for actual, expected in zip(actual_adjoints.values(), expected_adjoints.values()):
        assert_close(actual, expected)


@pytest.mark.parametrize('num_steps', list(range(3, 13)))
@pytest.mark.parametrize('sum_op,prod_op,state_domain', [
    (ops.add, ops.mul, bint(2)),
    (ops.add, ops.mul, bint(3)),
    (ops.logaddexp, ops.add, bint(2)),
    (ops.logaddexp, ops.add, bint(3)),
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
    # test mostly copied from test_sum_product.py
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

    # check forward pass (sanity check)
    assert_close(actual, expected, rtol=5e-4 * num_steps)

    # check backward pass
    for t, operand in enumerate(operands):
        actual_bwd_t = actual_bwd(time=t, prev="t_{}".format(t), curr="t_{}".format(t+1))
        expected_bwd = expected_bwds[operand].align(tuple(actual_bwd_t.inputs.keys()))
        check_funsor(actual_bwd_t, expected_bwd.inputs, expected_bwd.output)
        assert_close(actual_bwd_t, expected_bwd, rtol=5e-4 * num_steps)
