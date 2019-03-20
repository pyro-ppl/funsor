from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from six import integer_types

import torch

import funsor.ops as ops
from funsor.distributions import Gaussian, Delta
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import Finitary, optimize
from funsor.sum_product import _partition
from funsor.terms import Funsor, Number, Reduce, Variable, eager
from funsor.torch import Tensor


ATOMS = (Tensor, Gaussian, Delta, Number, Variable)


def _make_base_lhs(arg, reduced_vars, normalized=False):
    if not all(isinstance(d.dtype, integer_types) for d in arg.inputs.values()):
        raise NotImplementedError("TODO implement continuous base lhss")

    sizes = OrderedDict(set((var, dtype) for var, dtype in arg.inputs.items()))
    terms = tuple(
        Tensor(torch.ones((d.dtype,)) / float(d.dtype), OrderedDict([(var, d)]))
        if normalized else
        Tensor(torch.ones((d.dtype,)), OrderedDict([(var, d)]))
        for var, d in sizes.items() if var in reduced_vars
    )
    return Finitary(ops.mul, terms) if len(terms) > 1 else terms[0]


def _find_constants(lhs, operands, reduced_vars):
    constants, new_operands = [], []
    for operand in operands:
        if reduced_vars & frozenset(lhs.inputs) & frozenset(operand.inputs):
            new_operands.append(operand)
        else:
            constants.append(operand)
    return tuple(constants), tuple(new_operands)


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
    meas_vars = frozenset(lhs.inputs)
    int_vars = frozenset(rhs.inputs)
    assert reduced_vars <= meas_vars | int_vars
    progress = False
    if not reduced_vars <= meas_vars:
        rhs = rhs.reduce(ops.add, reduced_vars - meas_vars)
        reduced_vars = reduced_vars & meas_vars
        progress = True
    if not reduced_vars <= int_vars:
        lhs = lhs.reduce(ops.add, reduced_vars - int_vars)
        reduced_vars = reduced_vars & int_vars
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
        raise NotImplementedError("TODO implement subs")


@optimize.register(Contract, ATOMS, ATOMS, frozenset)
@eager.register(Contract, ATOMS, ATOMS, frozenset)
def contract_ground_ground(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

    return (lhs * rhs).reduce(ops.add, reduced_vars)


@optimize.register(Contract, Funsor, Reduce, frozenset)
def contract_reduce(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

    # XXX should we be doing this conversion at all given that Reduce
    # is already handled by the optimizer?
    if rhs.op is ops.add:
        base_lhs = _make_base_lhs(rhs.arg, rhs.reduced_vars, normalized=False)
        inner = Contract(base_lhs, rhs.arg, rhs.reduced_vars)
        return Contract(lhs, inner, reduced_vars)
    return None


@optimize.register(Contract, ATOMS, Finitary, frozenset)
def contract_ground_finitary(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

    # exploit linearity of integration
    if rhs.op is ops.add:
        return Finitary(
            ops.add,
            tuple(Contract(lhs, operand, reduced_vars) for operand in rhs.operands)
        )

    # pull out constant terms that do not depend on the lhs
    if rhs.op is ops.mul:
        constants, new_operands = _find_constants(lhs, rhs.operands, reduced_vars)
        if new_operands and constants:
            # this term should equal Finitary(mul, constants) for probability lhss
            outer = Finitary(ops.mul, constants)
            inner = Contract(lhs, Finitary(rhs.op, new_operands), reduced_vars)
            return outer * inner
        elif not new_operands and constants:
            return Finitary(ops.mul, constants)

    return None


@optimize.register(Contract, Finitary, (Finitary,) + ATOMS, frozenset)
def contract_finitary_ground(lhs, rhs, reduced_vars):
    result = _simplify_contract(lhs, rhs, reduced_vars)
    if result is not None:
        return result

    # recursively apply law of iterated expectation
    assert len(lhs.operands) > 1, "Finitary with one operand should have been passed through"
    if lhs.op is ops.mul:
        # TODO topologically order the lhs terms according to their variables
        # TODO don't reduce too early
        root_lhs, remaining_lhs = _order_lhss(lhs, reduced_vars)
        print(root_lhs, remaining_lhs)
        if remaining_lhs is not None:
            inner = Contract(remaining_lhs, rhs,
                             reduced_vars & frozenset(remaining_lhs.inputs))
            return Contract(root_lhs, inner,
                            reduced_vars & frozenset(root_lhs.inputs))

    return None


##############################
# utilities for testing follow
##############################

def naive_contract_einsum(eqn, *terms, **kwargs):
    """
    Use for testing Contract against einsum
    """
    assert "plates" not in kwargs

    backend = kwargs.pop('backend', 'torch')
    if backend in ('torch', 'pyro.ops.einsum.torch_log'):
        prod_op = ops.mul
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    inputs = inputs.split(',')
    assert len(inputs) == len(terms)
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs for d in inp)
    output_dims = frozenset(d for d in output)
    reduced_vars = input_dims - output_dims

    if backend == 'pyro.ops.einsum.torch_log':
        terms = tuple(term.exp() for term in terms)

    with interpretation(optimize):
        rhs = Finitary(prod_op, tuple(terms))
        lhs = _make_base_lhs(rhs, reduced_vars, normalized=False)
        assert frozenset(lhs.inputs) == reduced_vars
        # lhs = Number(1.)

        print("MEASURE: {}\n".format(lhs))
        print("INTEGRAND: {}\n".format(rhs))
        result = Contract(lhs, rhs, reduced_vars)
        print("RESULT: {}\n".format(result))

    if backend == 'pyro.ops.einsum.torch_log':
        result = result.log()

    return reinterpret(result)
