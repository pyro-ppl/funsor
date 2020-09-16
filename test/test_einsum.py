# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import opt_einsum
import pytest

import funsor
import funsor.ops as ops
from funsor.cnf import BACKEND_TO_EINSUM_BACKEND, BACKEND_TO_LOGSUMEXP_BACKEND, BACKEND_TO_MAP_BACKEND
from funsor.domains import Bint
from funsor.einsum import naive_einsum, naive_plated_einsum
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.tensor import Tensor
from funsor.terms import Variable, reflect
from funsor.testing import assert_close, make_einsum_example
from funsor.util import get_backend

EINSUM_EXAMPLES = [
    "a,b->",
    "ab,a->",
    "a,a->",
    "a,a->a",
    "a,a,a,ab->ab",
    "ab->ba",
    "ab,bc,cd->da",
]


def backend_to_einsum_backends(backend):
    backends = [BACKEND_TO_EINSUM_BACKEND[get_backend()],
                BACKEND_TO_LOGSUMEXP_BACKEND[get_backend()]]
    map_backend = BACKEND_TO_MAP_BACKEND[get_backend()]
    backends.append(map_backend)
    return backends


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', backend_to_einsum_backends(get_backend()))
def test_einsum(equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = opt_einsum.contract(equation, *operands, backend=backend)

    with interpretation(reflect):
        naive_ast = naive_einsum(equation, *funsor_operands, backend=backend)
        optimized_ast = apply_optimizer(naive_ast)
    print("Naive expression: {}".format(naive_ast))
    print("Optimized expression: {}".format(optimized_ast))
    actual_optimized = reinterpret(optimized_ast)  # eager by default
    actual = naive_einsum(equation, *funsor_operands, backend=backend)

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))
        actual_optimized = actual_optimized.align(tuple(outputs[0]))

    assert_close(actual, actual_optimized, atol=1e-4)
    assert expected.shape == actual.data.shape
    assert_close(expected, actual.data, rtol=1e-5, atol=1e-8)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.skipif(get_backend() == "numpy",
                    reason="funsor.distribution does not support numpy backend")
def test_einsum_categorical(equation):
    if get_backend() == "jax":
        from funsor.jax.distributions import Categorical
    else:
        from funsor.torch.distributions import Categorical

    inputs, outputs, sizes, operands, _ = make_einsum_example(equation)
    operands = [ops.abs(operand) / ops.abs(operand).sum(-1)[..., None]
                for operand in operands]

    expected = opt_einsum.contract(equation, *operands,
                                   backend=BACKEND_TO_EINSUM_BACKEND[get_backend()])

    with interpretation(reflect):
        funsor_operands = [
            Categorical(probs=Tensor(
                operand,
                inputs=OrderedDict([(d, Bint[sizes[d]]) for d in inp[:-1]])
            ))(value=Variable(inp[-1], Bint[sizes[inp[-1]]])).exp()
            for inp, operand in zip(inputs, operands)
        ]

        naive_ast = naive_einsum(equation, *funsor_operands)
        optimized_ast = apply_optimizer(naive_ast)

    print("Naive expression: {}".format(naive_ast))
    print("Optimized expression: {}".format(optimized_ast))
    actual_optimized = reinterpret(optimized_ast)  # eager by default
    actual = naive_einsum(equation, *map(reinterpret, funsor_operands))

    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))
        actual_optimized = actual_optimized.align(tuple(outputs[0]))

    assert_close(actual, actual_optimized, atol=1e-4)

    assert expected.shape == actual.data.shape
    assert_close(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


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


@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', [
    BACKEND_TO_EINSUM_BACKEND[get_backend()],
    BACKEND_TO_LOGSUMEXP_BACKEND[get_backend()],
    BACKEND_TO_MAP_BACKEND[get_backend()]
])
@pytest.mark.skipif(get_backend() != "torch",
                    reason="pyro.ops.contract.einsum does not work with numpy/jax backend.")
def test_plated_einsum(equation, plates, backend):
    from pyro.ops.contract import einsum as pyro_einsum

    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = pyro_einsum(equation, *operands, plates=plates, backend=backend, modulo_total=False)[0]
    with interpretation(reflect):
        naive_ast = naive_plated_einsum(equation, *funsor_operands, plates=plates, backend=backend)
        optimized_ast = apply_optimizer(naive_ast)
    actual_optimized = reinterpret(optimized_ast)  # eager by default
    actual = naive_plated_einsum(equation, *funsor_operands, plates=plates, backend=backend)

    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))
        actual_optimized = actual_optimized.align(tuple(outputs[0]))

    assert_close(actual, actual_optimized, atol=1e-3 if backend == 'torch' else 1e-4)

    assert expected.shape == actual.data.shape
    assert_close(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]
