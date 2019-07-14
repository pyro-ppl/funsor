from __future__ import absolute_import, division, print_function

import pytest

from funsor.distributions import Normal
from funsor.einsum import einsum, naive_plated_einsum
from funsor.interpreter import interpretation, reinterpret
from funsor.memoize import memoize
from funsor.terms import reflect
from funsor.testing import make_einsum_example, xfail_param


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
@pytest.mark.parametrize('same_lazy', [True, xfail_param(False, reason="issue w/ alpha conversion?")])
def test_einsum_complete_sharing(equation, plates, backend, einsum_impl, same_lazy):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        lazy_expr1 = einsum_impl(equation, *funsor_operands, backend=backend, plates=plates)
        lazy_expr2 = lazy_expr1 if same_lazy else \
            einsum_impl(equation, *funsor_operands, backend=backend, plates=plates)

    with memoize():
        expr1 = reinterpret(lazy_expr1)
        expr2 = reinterpret(lazy_expr2)
    expr3 = reinterpret(lazy_expr1)

    assert expr1 is expr2
    assert expr1 is not expr3


@pytest.mark.parametrize('equation,plates', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize('einsum_impl', [einsum, naive_plated_einsum])
@pytest.mark.parametrize('same_lazy', [True, xfail_param(False, reason="issue w/ alpha conversion?")])
def test_einsum_complete_sharing_reuse_cache(equation, plates, backend, einsum_impl, same_lazy):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        lazy_expr1 = einsum_impl(equation, *funsor_operands, backend=backend, plates=plates)
        lazy_expr2 = lazy_expr1 if same_lazy else \
            einsum_impl(equation, *funsor_operands, backend=backend, plates=plates)

    cache = {}
    with memoize(cache) as cache:
        expr1 = reinterpret(lazy_expr1)

    with memoize(cache):
        expr2 = reinterpret(lazy_expr2)

    expr3 = reinterpret(lazy_expr1)

    assert expr1 is expr2
    assert expr1 is not expr3


# @pytest.mark.xfail(reason="Joint and Joint.sample cannot directly be memoized in this way?")
def test_memoize_sample():

    with memoize():
        j1 = Normal(0, 1, 'x')
        j2 = Normal(0, 1, 'x')
        x1 = j1.sample(frozenset({'x'}))
        x12 = j1.sample(frozenset({'x'}))
        x2 = j2.sample(frozenset({'x'}))

    assert j1 is j2
    assert x1 is x12
    assert x1 is x2
