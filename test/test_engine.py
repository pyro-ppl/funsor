from __future__ import absolute_import, division, print_function

import pytest
import torch

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.engine.contract_engine import eval as _contract_eval
from funsor.engine.materialize import materialize
from funsor.engine.opteinsum_engine import eval as _opteinsum_eval
from funsor.engine.tree_engine import eval as _tree_eval


def xfail_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


def opteinsum_eval(x): return _opteinsum_eval(x)


def contract_eval(x): return _contract_eval(x)  # for pytest param naming


def tree_eval(x): return _tree_eval(x)  # for pytest param naming


@pytest.mark.parametrize('eval', [opteinsum_eval, contract_eval])
@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
def test_mm(eval, materialize_f, materialize_g):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = materialize(f)

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_g:
        g = materialize(g)

    h = (f * g).sum('j')
    eval_h = eval(h)
    assert isinstance(eval_h, funsor.Tensor)
    assert eval_h.dims == h.dims
    assert eval_h.shape == h.shape
    for i in range(3):
        for k in range(5):
            assert eval_h[i, k] == funsor.materialize(h[i, k])


@pytest.mark.parametrize('eval', [opteinsum_eval, contract_eval])
@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
def test_logsumproductexp(eval, materialize_f, materialize_g):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = funsor.materialize(f)

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_g:
        g = funsor.materialize(g)

    log_prob = funsor.Tensor(('log_prob',), torch.randn(10))
    h = (log_prob[f] + log_prob[g]).logsumexp('j')

    eval_h = eval(h)
    assert isinstance(eval_h, funsor.Tensor)
    assert eval_h.dims == h.dims
    assert eval_h.shape == h.shape
    for i in range(3):
        for k in range(5):
            assert (eval_h[i, k] - funsor.materialize(h[i, k])) < 1e-6


@pytest.mark.parametrize('eval', [
    opteinsum_eval,
    contract_eval,
])
def test_hmm_discrete_gaussian(eval):
    hidden_dim = 2
    num_steps = 3
    trans = funsor.Tensor(('prev', 'curr'), torch.tensor([[0.9, 0.1], [0.1, 0.9]]).log())
    locs = funsor.Tensor(('state',), torch.randn(hidden_dim))
    emit = dist.Normal(loc=locs, scale=1.)
    assert emit.dims == ('value', 'state')
    data = funsor.Tensor(('t',), torch.randn(num_steps))

    log_prob = 0.
    x_curr = 0
    for t, y in enumerate(data):
        x_prev, x_curr = x_curr, funsor.Variable('x_{}'.format(t), hidden_dim)
        log_prob += trans(prev=x_prev, curr=x_curr)
        log_prob += emit(state=x_curr, value=y)
    log_prob = log_prob.reduce(ops.logaddexp)
    log_prob = eval(log_prob)
    assert isinstance(log_prob, funsor.Tensor)
    assert not log_prob.dims


@pytest.mark.parametrize('num_steps', [1, 2, 3])
@pytest.mark.parametrize('eval', [
    xfail_param(opteinsum_eval, reason='bad trampoline?'),
    xfail_param(contract_eval, reason='cannot match Substitution(Normal)'),
    xfail_param(tree_eval, reason='incomplete Normal-Normal math'),
])
def test_hmm_gaussian_gaussian(eval, num_steps):
    trans = dist.Normal(funsor.Variable('prev', 'real'), 0.1)
    emit = dist.Normal(funsor.Variable('state', 'real'), 1.)
    assert emit.dims == ('value', 'state')
    data = funsor.Tensor(('t',), torch.randn(num_steps))

    log_prob = 0.
    x_curr = 0.
    for t, y in enumerate(data):
        x_prev, x_curr = x_curr, funsor.Variable('x_{}'.format(t), 'real')
        log_prob += trans(prev=x_prev, value=x_curr)
        log_prob += emit(state=x_curr, value=y)
    log_prob = log_prob.reduce(ops.logaddexp)
    log_prob = eval(log_prob)
    assert isinstance(log_prob, funsor.Tensor)
    assert not log_prob.dims
