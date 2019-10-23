import pytest
import torch

import funsor.ops as ops
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
@pytest.mark.parametrize('einsum_impl,same_lazy', [
    (einsum, True),
    (einsum, xfail_param(False, reason="nested interpreters?")),
    (naive_plated_einsum, True),
    (naive_plated_einsum, False)
])
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
@pytest.mark.parametrize('einsum_impl,same_lazy', [
    (einsum, True),
    (einsum, xfail_param(False, reason="nested interpreters?")),
    (naive_plated_einsum, True),
    (naive_plated_einsum, False)
])
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


@pytest.mark.parametrize('check_sample', [
    False, xfail_param(True, reason="Joint.sample cannot directly be memoized in this way yet")])
def test_memoize_sample(check_sample):

    with memoize():
        m, s = torch.tensor(0.), torch.tensor(1.)
        j1 = Normal(m, s, 'x')
        j2 = Normal(m, s, 'x')
        x1 = j1.sample(frozenset({'x'}))
        x12 = j1.sample(frozenset({'x'}))
        x2 = j2.sample(frozenset({'x'}))

    # this assertion now passes
    assert j1 is j2

    # these assertions fail because sample is not memoized
    if check_sample:
        assert x1 is x12
        assert x1 is x2


@pytest.mark.parametrize("eqn1,eqn2", [("ab,bc,cd->d", "de,ef,fg->")])
@pytest.mark.parametrize("einsum_impl1", [naive_plated_einsum, xfail_param(einsum, reason="nested interpreters?")])
@pytest.mark.parametrize("einsum_impl2", [naive_plated_einsum, xfail_param(einsum, reason="nested interpreters?")])
@pytest.mark.parametrize('backend1', ['torch', 'pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize('backend2', ['torch', 'pyro.ops.einsum.torch_log'])
def test_nested_einsum_complete_sharing(eqn1, eqn2, einsum_impl1, einsum_impl2, backend1, backend2):

    inputs1, outputs1, sizes1, operands1, funsor_operands1 = make_einsum_example(eqn1, sizes=(3,))
    inputs2, outputs2, sizes2, operands2, funsor_operands2 = make_einsum_example(eqn2, sizes=(3,))

    with memoize():
        output1_1 = einsum_impl1(eqn1, *funsor_operands1, backend=backend1)
        output2_1 = einsum_impl2(outputs1[0] + "," + eqn2, *([output1_1] + funsor_operands2), backend=backend2)

        output1_2 = einsum_impl1(eqn1, *funsor_operands1, backend=backend1)
        output2_2 = einsum_impl2(outputs1[0] + "," + eqn2, *([output1_2] + funsor_operands2), backend=backend2)

    assert output1_1 is output1_2
    assert output2_1 is output2_2


def test_nested_complete_sharing_direct():

    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example("ab,bc,cd->d")
    ab, bc, cd = funsor_operands

    # avoids the complicated internal interpreter usage of the nested optimized einsum tests above
    with interpretation(reflect):
        c1 = (ab * bc).reduce(ops.add, frozenset({"a", "b"}))
        d1 = (c1 * cd).reduce(ops.add, frozenset({"c"}))

        # this does not trigger a second alpha-renaming
        c2 = (ab * bc).reduce(ops.add, frozenset({"a", "b"}))
        d2 = (c2 * cd).reduce(ops.add, frozenset({"c"}))

    with memoize():
        assert reinterpret(c1) is reinterpret(c2)
        assert reinterpret(d1) is reinterpret(d2)
