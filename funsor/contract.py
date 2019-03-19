from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from six import integer_types

import torch

import funsor.ops as ops
from funsor.distributions import Gaussian, Delta
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import Finitary, optimize
from funsor.terms import Funsor, Number, Reduce, Variable, eager
from funsor.torch import Tensor


ATOMS = (Tensor, Gaussian, Delta, Number, Variable)


def _make_base_measure(arg, reduced_vars, normalized=False):
    if not all(isinstance(d.dtype, integer_types) for d in arg.inputs.values()):
        raise NotImplementedError("TODO implement continuous base measures")

    sizes = OrderedDict(set((var, dtype) for var, dtype in arg.inputs.items()))
    terms = tuple(
        Tensor(torch.ones((d.dtype,)) / float(d.dtype), OrderedDict([(var, d)]))
        if normalized else
        Tensor(torch.ones((d.dtype,)), OrderedDict([(var, d)]))
        for var, d in sizes.items() if var in reduced_vars
    )
    return Finitary(ops.mul, terms) if len(terms) > 1 else terms[0]


def _find_constants(measure, operands, reduced_vars):
    constants, new_operands = [], []
    for operand in operands:
        if reduced_vars & frozenset(measure.inputs) & frozenset(operand.inputs):
            new_operands.append(operand)
        else:
            constants.append(operand)
    return tuple(constants), tuple(new_operands)


def _order_measures(measure, reduced_vars):
    assert isinstance(measure, Finitary)

    # old behavior (incorrect?)
    root_measure = measure.operands[0]
    remaining_measure = Finitary(measure.op, measure.operands[1:])

    # new behavior (suboptimal; linear (?) in number of integrands at each step...)
    # terms = measure.operands
    # term_to_terms = {
    #     term: set(t for t in terms if set(t.inputs) & set(term.inputs))
    #     for term in terms
    # }
    # while terms:
    #     term = terms.pop()
    #     new_terms.append(term)
    #     for child_term in term_to_terms[term]:
    #         pass

    return root_measure, remaining_measure


def _simplify_contract(measure, integrand, reduced_vars):
    """
    Reduce free variables that do not appear explicitly in the measure
    """
    meas_vars = frozenset(measure.inputs)
    int_vars = frozenset(integrand.inputs)
    assert reduced_vars <= meas_vars | int_vars
    progress = False
    if not reduced_vars <= meas_vars:
        integrand = integrand.reduce(ops.add, reduced_vars - meas_vars)
        reduced_vars = reduced_vars & meas_vars
        progress = True
    if not reduced_vars <= int_vars:
        measure = measure.reduce(ops.add, reduced_vars - int_vars)
        reduced_vars = reduced_vars & int_vars
        progress = True

    if progress:
        return Contract(measure, integrand, reduced_vars)

    return None


class Contract(Funsor):

    def __init__(self, measure, integrand, reduced_vars):
        assert isinstance(measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, frozenset)
        inputs = OrderedDict([(k, d) for t in (measure, integrand)
                              for k, d in t.inputs.items() if k not in reduced_vars])
        output = integrand.output
        super(Contract, self).__init__(inputs, output)
        self.measure = measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def eager_subs(self, subs):
        raise NotImplementedError("TODO implement subs")


@optimize.register(Contract, ATOMS, ATOMS, frozenset)
@eager.register(Contract, ATOMS, ATOMS, frozenset)
def contract_ground_ground(measure, integrand, reduced_vars):
    result = _simplify_contract(measure, integrand, reduced_vars)
    if result is not None:
        return result

    return (measure * integrand).reduce(ops.add, reduced_vars)


@optimize.register(Contract, Funsor, Reduce, frozenset)
def contract_reduce(measure, integrand, reduced_vars):
    result = _simplify_contract(measure, integrand, reduced_vars)
    if result is not None:
        return result

    # XXX should we be doing this conversion at all given that Reduce
    # is already handled by the optimizer?
    if integrand.op is ops.add:
        base_measure = _make_base_measure(integrand.arg, integrand.reduced_vars, normalized=False)
        inner = Contract(base_measure, integrand.arg, integrand.reduced_vars)
        return Contract(measure, inner, reduced_vars)
    return None


@optimize.register(Contract, ATOMS, Finitary, frozenset)
def contract_ground_finitary(measure, integrand, reduced_vars):
    result = _simplify_contract(measure, integrand, reduced_vars)
    if result is not None:
        return result

    # exploit linearity of integration
    if integrand.op is ops.add:
        return Finitary(
            ops.add,
            tuple(Contract(measure, operand, reduced_vars) for operand in integrand.operands)
        )

    # pull out constant terms that do not depend on the measure
    if integrand.op is ops.mul:
        constants, new_operands = _find_constants(measure, integrand.operands, reduced_vars)
        if new_operands and constants:
            # this term should equal Finitary(mul, constants) for probability measures
            outer = Finitary(ops.mul, constants)
            inner = Contract(measure, Finitary(integrand.op, new_operands), reduced_vars)
            return outer * inner
        elif not new_operands and constants:
            return Finitary(ops.mul, constants)

    return None


@optimize.register(Contract, Finitary, (Finitary,) + ATOMS, frozenset)
def contract_finitary_ground(measure, integrand, reduced_vars):
    result = _simplify_contract(measure, integrand, reduced_vars)
    if result is not None:
        return result

    # recursively apply law of iterated expectation
    assert len(measure.operands) > 1, "Finitary with one operand should have been passed through"
    if measure.op is ops.mul:
        # TODO topologically order the measure terms according to their variables
        root_measure, remaining_measure = _order_measures(measure, reduced_vars)
        if remaining_measure is not None:
            inner = Contract(remaining_measure, integrand,
                             reduced_vars & frozenset(remaining_measure.inputs))
            return Contract(root_measure, inner,
                            reduced_vars & frozenset(root_measure.inputs))

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
        integrand = Finitary(prod_op, tuple(terms))
        measure = _make_base_measure(integrand, reduced_vars, normalized=False)
        assert frozenset(measure.inputs) == reduced_vars
        # measure = Number(1.)

        print("MEASURE: {}\n".format(measure))
        print("INTEGRAND: {}\n".format(integrand))
        result = Contract(measure, integrand, reduced_vars)
        print("RESULT: {}\n".format(result))

    if backend == 'pyro.ops.einsum.torch_log':
        result = result.log()

    return reinterpret(result)
