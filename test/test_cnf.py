import pytest

from funsor.cnf import Contraction
from funsor.einsum import einsum, naive_plated_einsum
from funsor.interpreter import interpretation, reinterpret
from funsor.terms import Number, eager, normalize, reflect
from funsor.testing import assert_close, check_funsor, make_einsum_example  # , xfail_param
from funsor.torch import Tensor


EINSUM_EXAMPLES = [
    ("a,b->", ''),
    ("ab,a->", ''),
    ("a,a->", ''),
    ("a,a->a", ''),
    ("ab,bc,cd->da", ''),
    ("ab,cd,bc->da", ''),
    ("a,a,a,ab->ab", ''),
    ('i->', 'i'),
    (',i->', 'i'),
    ('ai->', 'i'),
    (',ai,abij->', 'ij'),
    ('a,ai,bij->', 'ij'),
    ('ai,abi,bci,cdi->', 'i'),
    ('aij,abij,bcij->', 'ij'),
    ('a,abi,bcij,cdij->', 'ij'),
]


@pytest.mark.parametrize('equation,plates', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize('einsum_impl', [einsum, naive_plated_einsum])
def test_normalize_einsum(equation, plates, backend, einsum_impl):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        expr = einsum_impl(equation, *funsor_operands, backend=backend, plates=plates)

    with interpretation(normalize):
        transformed_expr = reinterpret(expr)

    assert isinstance(transformed_expr, Contraction)
    check_funsor(transformed_expr, expr.inputs, expr.output)

    if plates:
        assert all(isinstance(v, (Number, Tensor, Contraction)) for v in transformed_expr.terms)
    else:
        assert all(isinstance(v, (Number, Tensor)) for v in transformed_expr.terms)

    with interpretation(normalize):
        transformed_expr2 = reinterpret(transformed_expr)

    assert transformed_expr2 is transformed_expr  # check normalization

    with interpretation(eager):
        actual = reinterpret(transformed_expr)
        expected = reinterpret(expr)

    assert_close(actual, expected, rtol=1e-4)
