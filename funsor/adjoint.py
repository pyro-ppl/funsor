from __future__ import absolute_import, division, print_function

from collections import defaultdict
import torch

import funsor.ops as ops
from funsor.interpreter import interpretation, reinterpret
from funsor.ops import AssociativeOp
from funsor.registry import KeyedRegistry
from funsor.terms import Binary, Funsor, Number, Reduce, Variable, eager
from funsor.torch import Tensor


class AdjointTape(object):

    def __init__(self):
        self.tape = []

    def __call__(self, cls, *args):
        result = eager(cls, *args)
        if cls in (Reduce, Binary, Tensor):
            self.tape.append((result, cls, args))
        return result


def adjoint(expr, targets, start=Number(0.)):

    adjoint_values = defaultdict(lambda: Number(0.))  # 1 in logspace
    multiplicities = defaultdict(lambda: 0)

    tape_recorder = AdjointTape()
    with interpretation(tape_recorder):
        adjoint_values[reinterpret(expr)] = start

    while tape_recorder.tape:
        output, fn, inputs = tape_recorder.tape.pop()
        in_adjs = adjoint_ops(fn, adjoint_values[output], output, *inputs)
        for v, adjv in in_adjs.items():
            multiplicities[v] += 1
            adjoint_values[v] = adjoint_values[v] + adjv  # product in logspace

    target_adjs = {}
    for v in targets:
        target_adjs[v] = adjoint_values[v] / multiplicities[v]
        if not isinstance(v, Variable):
            target_adjs[v] = target_adjs[v] + v
    return target_adjs


# logaddexp/add
def _fail_default(*args):
    raise NotImplementedError("Should not be here! {}".format(args))


adjoint_ops = KeyedRegistry(default=_fail_default)


@adjoint_ops.register(Tensor, Funsor, Funsor, torch.Tensor, tuple, object)
def adjoint_tensor(out_adj, out, data, inputs, dtype):
    all_vars = frozenset(k for (k, v) in inputs)
    in_adjs = {}
    for (k, v) in inputs:
        in_adj = (out_adj + out).reduce(ops.logaddexp, all_vars - {k})
        in_adjs[Variable(k, v)] = in_adj
    return in_adjs


@adjoint_ops.register(Binary, Funsor, Funsor, AssociativeOp, Funsor, Funsor)
def adjoint_binary(out_adj, out, op, lhs, rhs):
    assert op is ops.add

    lhs_reduced_vars = frozenset(rhs.inputs) - frozenset(lhs.inputs)
    lhs_adj = (out_adj + rhs).reduce(ops.logaddexp, lhs_reduced_vars)

    rhs_reduced_vars = frozenset(lhs.inputs) - frozenset(rhs.inputs)
    rhs_adj = (out_adj + lhs).reduce(ops.logaddexp, rhs_reduced_vars)

    return {lhs: lhs_adj, rhs: rhs_adj}


@adjoint_ops.register(Reduce, Funsor, Funsor, AssociativeOp, Funsor, frozenset)
def adjoint_reduce(out_adj, out, op, arg, reduced_vars):
    assert op in (ops.logaddexp, ops.add)

    if op is ops.logaddexp:
        return {arg: out_adj + (arg * 0.)}  # XXX hack to simulate "expand"
    elif op is ops.add:  # plate!
        return {arg: out_adj + Binary(ops.safesub, out, arg)}
