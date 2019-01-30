from __future__ import absolute_import, division, print_function

import itertools

import pytest

import funsor


def test_materialize():

    @funsor.lazy(3, 4)
    def f(i, j):
        return i + j

    g = f.materialize()

    assert g.dims == f.dims
    assert g.shape == f.shape
    for key in itertools.product(*map(funsor.domain, g.shape)):
        assert f[key] == g[key]


@pytest.mark.parametrize("g_lazy", [True, False], ids=["lazy", "torch"])
@pytest.mark.parametrize("f_lazy", [True, False], ids=["lazy", "torch"])
def test_mm_fn_fn(f_lazy, g_lazy):

    @funsor.lazy(3, 4)
    def f(i, j):
        return i + j

    assert f.dims == ("i", "j")
    assert f.shape == (3, 4)

    @funsor.lazy(4, 5)
    def g(j, k):
        return j + k

    assert g.dims == ("j", "k")
    assert g.shape == (4, 5)

    h = f.mm(g)
    assert h.dims == ("i", "k")
    assert h.shape == (3, 5)
    for i in range(3):
        for k in range(5):
            assert h[i, k] == sum(f[i, j] * g[j, k] for j in range(4))
