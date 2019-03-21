from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import funsor.ops as ops
from funsor.optimizer import Finitary, optimize
from funsor.sum_product import _partition
from funsor.terms import Funsor, eager


def _order_lhss(lhs, reduced_vars):
    assert isinstance(lhs, Finitary)

    components = _partition(lhs.operands, reduced_vars)
    root_lhs = Finitary(ops.mul, tuple(components[0][0]))
    if len(components) > 1:
        remaining_lhs = Finitary(ops.mul, tuple(t for c in components[1:] for t in c[0]))
    else:
        remaining_lhs = None

    return root_lhs, remaining_lhs


def _simplify_contract(fn, lhs, rhs, reduced_vars):
    """
    Reduce free variables that do not appear explicitly in the lhs
    """
    if not reduced_vars:
        return lhs * rhs

    lhs_vars = frozenset(lhs.inputs)
    rhs_vars = frozenset(rhs.inputs)
    assert reduced_vars <= lhs_vars | rhs_vars
    progress = False
    if not reduced_vars <= lhs_vars:
        rhs = rhs.reduce(ops.add, reduced_vars - lhs_vars)
        reduced_vars = reduced_vars & lhs_vars
        progress = True
    if not reduced_vars <= rhs_vars:
        lhs = lhs.reduce(ops.add, reduced_vars - rhs_vars)
        reduced_vars = reduced_vars & rhs_vars
        progress = True
    if progress:
        return Contract(lhs, rhs, reduced_vars)

    return fn(lhs, rhs, reduced_vars)


def contractor(fn):
    """
    Decorator for contract implementations to simplify inputs.
    """
    return functools.partial(_simplify_contract, fn)


class Contract(Funsor):

    def __init__(self, lhs, rhs, reduced_vars):
        assert isinstance(lhs, Funsor)
        assert isinstance(rhs, Funsor)
        assert isinstance(reduced_vars, frozenset)
        inputs = OrderedDict([(k, d) for t in (lhs, rhs)
                              for k, d in t.inputs.items() if k not in reduced_vars])
        output = rhs.output
        super(Contract, self).__init__(inputs, output)
        self.lhs = lhs
        self.rhs = rhs
        self.reduced_vars = reduced_vars

    def eager_subs(self, subs):
        # basically copied from Reduce.eager_subs
        subs = tuple((k, v) for k, v in subs if k not in self.reduced_vars)
        if not any(k in self.inputs for k, v in subs):
            return self
        if not all(self.reduced_vars.isdisjoint(v.inputs) for k, v in subs):
            raise NotImplementedError('TODO alpha-convert to avoid conflict')
        return Contract(self.lhs.eager_subs(subs), self.rhs.eager_subs(subs),
                        self.reduced_vars)


@optimize.register(Contract, Funsor, Funsor, frozenset)
@eager.register(Contract, Funsor, Funsor, frozenset)
@contractor
def contract_funsor_funsor(lhs, rhs, reduced_vars):
    return (lhs * rhs).reduce(ops.add, reduced_vars)


@optimize.register(Contract, Funsor, Finitary, frozenset)
@contractor
def contract_ground_finitary(lhs, rhs, reduced_vars):
    return Contract(rhs, lhs, reduced_vars)


@optimize.register(Contract, Finitary, (Finitary, Funsor), frozenset)
@contractor
def contract_finitary_ground(lhs, rhs, reduced_vars):
    # exploit linearity of contraction
    if lhs.op is ops.add:
        return Finitary(
            ops.add,
            tuple(Contract(operand, rhs, reduced_vars) for operand in lhs.operands)
        )

    # recursively apply law of iterated expectation
    assert len(lhs.operands) > 1, "Finitary with one operand should have been passed through"
    if lhs.op is ops.mul:
        root_lhs, remaining_lhs = _order_lhss(lhs, reduced_vars)
        if remaining_lhs is not None:
            inner = Contract(remaining_lhs, rhs,
                             reduced_vars & frozenset(remaining_lhs.inputs))
            return Contract(root_lhs, inner,
                            reduced_vars & frozenset(root_lhs.inputs))

    return None


__all__ = [
    'Contract',
    'contractor',
]
