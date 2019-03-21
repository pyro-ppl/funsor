from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import opt_einsum

import funsor.ops as ops
from funsor.distributions import Gaussian, Delta
from funsor.optimizer import Finitary, optimize
from funsor.sum_product import _partition
from funsor.terms import Funsor, Number, Variable, eager
from funsor.torch import Tensor


# TODO handle Joint as well
ATOMS = (Tensor, Gaussian, Delta, Number, Variable)


def _order_lhss(lhs, reduced_vars):
    assert isinstance(lhs, Finitary)

    components = _partition(lhs.operands, reduced_vars)
    root_lhs = Finitary(ops.mul, tuple(components[0][0]))
    if len(components) > 1:
        remaining_lhs = Finitary(ops.mul, tuple(t for c in components[1:] for t in c[0]))
    else:
        remaining_lhs = None

    return root_lhs, remaining_lhs


def _simplify_contract(lhs, rhs, reduced_vars):
    """
    Reduce free variables that do not appear explicitly in the lhs
    """
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

    return None


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


@optimize.register(Contract, ATOMS[1:], ATOMS, frozenset)
@optimize.register(Contract, ATOMS, ATOMS[1:], frozenset)
@eager.register(Contract, ATOMS[1:], ATOMS, frozenset)
@eager.register(Contract, ATOMS, ATOMS[1:], frozenset)
def contract_ground_ground(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

    return (lhs * rhs).reduce(ops.add, reduced_vars)


@eager.register(Contract, Tensor, Tensor, frozenset)
def eager_contract_tensor_tensor(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

    out_inputs = OrderedDict([(k, d) for t in (lhs, rhs)
                              for k, d in t.inputs.items() if k not in reduced_vars])

    return Tensor(
        opt_einsum.contract(lhs.data, list(lhs.inputs.keys()),
                            rhs.data, list(rhs.inputs.keys()),
                            list(out_inputs.keys()), backend="torch"),
        out_inputs
    )


@optimize.register(Contract, ATOMS, Finitary, frozenset)
def contract_ground_finitary(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

    return Contract(rhs, lhs, reduced_vars)


@optimize.register(Contract, Finitary, (Finitary,) + ATOMS, frozenset)
def contract_finitary_ground(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

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
