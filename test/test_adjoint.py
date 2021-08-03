# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor
import funsor.ops as ops
from funsor.adjoint import AdjointTape
from funsor.domains import Bint
from funsor.einsum import BACKEND_ADJOINT_OPS, einsum, naive_einsum, naive_plated_einsum
from funsor.optimizer import apply_optimizer
from funsor.sum_product import (
    MarkovProduct,
    naive_sequential_sum_product,
    sequential_sum_product,
    sum_product,
)
from funsor.terms import Variable, reflect
from funsor.testing import (
    assert_close,
    make_einsum_example,
    make_plated_hmm_einsum,
    random_gaussian,
    random_tensor,
    xfail_param,
)
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "torch",
    reason="numpy/jax backend requires porting pyro.ops.einsum",
)
if get_backend() == "torch":
    import torch
    from pyro.ops.contract import einsum as pyro_einsum
    from pyro.ops.einsum.adjoint import require_backward as pyro_require_backward

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


@pytest.mark.parametrize("einsum_impl", [naive_einsum, einsum])
@pytest.mark.parametrize("equation", EINSUM_EXAMPLES)
@pytest.mark.parametrize(
    "backend",
    [
        "pyro.ops.einsum.torch_marginal",
        xfail_param("pyro.ops.einsum.torch_map", reason="wrong adjoint"),
    ],
)
def test_einsum_adjoint(einsum_impl, equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    with AdjointTape() as tape:
        fwd_expr = einsum_impl(equation, *funsor_operands, backend=backend)
    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(equation, *operands, modulo_total=True, backend=backend)[
        0
    ]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        # actual = actuals[fv]
        actual = prod_op(actuals[fv], fv).reduce(
            sum_op, actuals[fv].input_vars - fv.input_vars
        )
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


PLATED_EINSUM_EXAMPLES = [
    (",i->", "i"),
    ("i->", "i"),
    ("ai->", "i"),
    (",ai,abij->", "ij"),
    ("a,ai,bij->", "ij"),
    ("ai,abi,bci,cdi->", "i"),
    ("aij,abij,bcij->", "ij"),
    ("a,abi,bcij,cdij->", "ij"),
]


@pytest.mark.parametrize("einsum_impl", [naive_plated_einsum, einsum])
@pytest.mark.parametrize("equation,plates", PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize(
    "backend",
    [
        "pyro.ops.einsum.torch_marginal",
        xfail_param("pyro.ops.einsum.torch_map", reason="wrong adjoint"),
    ],
)
def test_plated_einsum_adjoint(einsum_impl, equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    with AdjointTape() as tape:
        fwd_expr = einsum_impl(
            equation, *funsor_operands, plates=plates, backend=backend
        )
    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(
        equation, *operands, modulo_total=False, plates=plates, backend=backend
    )[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        actual = prod_op(actuals[fv], fv).reduce(
            sum_op, actuals[fv].input_vars - fv.input_vars
        )
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


OPTIMIZED_PLATED_EINSUM_EXAMPLES = [
    make_plated_hmm_einsum(num_steps, num_obs_plates=b, num_hidden_plates=a)
    for num_steps in [20, 30, 50]
    for (a, b) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
]


@pytest.mark.parametrize("equation,plates", OPTIMIZED_PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize(
    "backend",
    [
        "pyro.ops.einsum.torch_marginal",
        xfail_param("pyro.ops.einsum.torch_map", reason="wrong adjoint"),
    ],
)
def test_optimized_plated_einsum_adjoint(equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    sum_op, prod_op = BACKEND_ADJOINT_OPS[backend]

    with AdjointTape() as tape:
        fwd_expr = einsum(equation, *funsor_operands, plates=plates, backend=backend)
    actuals = tape.adjoint(sum_op, prod_op, fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(
        equation, *operands, modulo_total=False, plates=plates, backend=backend
    )[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        actual = prod_op(actuals[fv], fv).reduce(
            sum_op, actuals[fv].input_vars - fv.input_vars
        )
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


@pytest.mark.parametrize("num_steps", list(range(3, 13)))
@pytest.mark.parametrize(
    "sum_op,prod_op,state_domain",
    [
        (ops.add, ops.mul, Bint[2]),
        (ops.add, ops.mul, Bint[3]),
        (ops.logaddexp, ops.add, Bint[2]),
        (ops.logaddexp, ops.add, Bint[3]),
        # (ops.logaddexp, ops.add, Real),
        # (ops.logaddexp, ops.add, Reals[2]),
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
        xfail_param(MarkovProduct, reason="mysteriously doubles adjoint values?"),
    ],
)
def test_sequential_sum_product_adjoint(
    impl, sum_op, prod_op, batch_inputs, state_domain, num_steps
):
    # test mostly copied from test_sum_product.py
    inputs = OrderedDict(batch_inputs)
    inputs.update(prev=state_domain, curr=state_domain)
    inputs["time"] = Bint[num_steps]
    if state_domain.dtype == "real":
        trans = random_gaussian(inputs)
    else:
        trans = random_tensor(inputs)
    time = Variable("time", Bint[num_steps])

    with AdjointTape() as actual_tape:
        actual = impl(sum_op, prod_op, trans, time, {"prev": "curr"})
        actual = actual.reduce(sum_op)

    # Check against contract.
    operands = tuple(
        trans(time=t, prev="t_{}".format(t), curr="t_{}".format(t + 1))
        for t in range(num_steps)
    )
    reduce_vars = frozenset("t_{}".format(t) for t in range(1, num_steps))
    with AdjointTape() as expected_tape:
        with reflect:
            lazy_expected = sum_product(sum_op, prod_op, operands, reduce_vars)
        expected = apply_optimizer(lazy_expected)
        expected = expected.reduce(sum_op)

    # check forward pass (sanity check)
    assert_close(
        actual, expected.align(tuple(actual.inputs.keys())), rtol=5e-3 * num_steps
    )

    # perform backward passes only after the sanity check
    expected_bwds = expected_tape.adjoint(sum_op, prod_op, expected, operands)
    actual_bwd = actual_tape.adjoint(sum_op, prod_op, actual, (trans,))[trans]

    # check backward pass
    for t, operand in enumerate(operands):
        actual_bwd_t = actual_bwd(
            time=t, prev="t_{}".format(t), curr="t_{}".format(t + 1)
        )
        expected_bwd = expected_bwds[operand]
        assert (actual_bwd_t - expected_bwd).abs().data.max() < 5e-3 * num_steps
