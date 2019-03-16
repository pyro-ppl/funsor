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


def _make_base_measure(arg, reduced_vars, normalized=True):
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


def _simplify_integrate(measure, integrand, reduced_vars):
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
        return Integrate(measure, integrand, reduced_vars)

    return None


class Integrate(Funsor):

    def __init__(self, measure, integrand, reduced_vars):
        assert isinstance(measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(reduced_vars, frozenset)
        inputs = OrderedDict([(k, d) for t in (measure, integrand)
                              for k, d in t.inputs.items() if k not in reduced_vars])
        output = integrand.output
        super(Integrate, self).__init__(inputs, output)
        self.measure = measure
        self.integrand = integrand
        self.reduced_vars = reduced_vars

    def eager_subs(self, subs):
        raise NotImplementedError("TODO implement subs")


@optimize.register(Integrate, ATOMS, ATOMS, frozenset)
@eager.register(Integrate, ATOMS, ATOMS, frozenset)
def integrate_ground_ground(measure, integrand, reduced_vars):
    result = _simplify_integrate(measure, integrand, reduced_vars)
    if result is not None:
        return result

    return (measure * integrand).reduce(ops.add, reduced_vars)


@optimize.register(Integrate, Funsor, Reduce, frozenset)
def integrate_reduce(measure, integrand, reduced_vars):
    result = _simplify_integrate(measure, integrand, reduced_vars)
    if result is not None:
        return result

    # XXX should we be doing this conversion at all given that Reduce
    # is already handled by the optimizer?
    if integrand.op is ops.add:
        base_measure = _make_base_measure(integrand.arg, integrand.reduced_vars, normalized=False)
        inner = Integrate(base_measure, integrand.arg, integrand.reduced_vars)
        return Integrate(measure, inner, reduced_vars)
    return None


@optimize.register(Integrate, ATOMS, Finitary, frozenset)
def integrate_ground_finitary(measure, integrand, reduced_vars):
    result = _simplify_integrate(measure, integrand, reduced_vars)
    if result is not None:
        return result

    # exploit linearity of integration
    if integrand.op is ops.add:
        return Finitary(
            ops.add,
            tuple(Integrate(measure, operand, reduced_vars) for operand in integrand.operands)
        )

    # pull out constant terms that do not depend on the measure
    if integrand.op is ops.mul:
        constants, new_operands = _find_constants(measure, integrand.operands, reduced_vars)
        if new_operands and constants:
            # this term should equal Finitary(mul, constants) for probability measures
            # outer = Integrate(measure, Finitary(integrand.op, constants), reduced_vars)
            outer = Finitary(ops.mul, tuple(Integrate(measure, c, reduced_vars) for c in constants))
            inner = Integrate(measure, Finitary(integrand.op, new_operands), reduced_vars)
            return outer * inner
        elif not new_operands and constants:
            return Finitary(ops.mul, tuple(Integrate(measure, c, reduced_vars) for c in constants))

    return None


@optimize.register(Integrate, Finitary, (Finitary,) + ATOMS, frozenset)
def integrate_finitary_ground(measure, integrand, reduced_vars):
    result = _simplify_integrate(measure, integrand, reduced_vars)
    if result is not None:
        return result

    # recursively apply law of iterated expectation
    assert len(measure.operands) > 1, "Finitary with one operand should have been passed through"
    if measure.op is ops.mul:
        # TODO topologically order the measure terms according to their variables
        root_measure = measure.operands[0]
        remaining_measure = Finitary(measure.op, measure.operands[1:])
        inner = Integrate(remaining_measure, integrand, reduced_vars & frozenset(remaining_measure.inputs))
        return Integrate(root_measure, inner, reduced_vars & frozenset(root_measure.inputs))

    return None


##############################
# utilities for testing follow
##############################

def naive_integrate_einsum(eqn, *terms, **kwargs):
    """
    Use for testing Integrate against einsum
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
        # measure = _make_base_measure(integrand, reduced_vars, normalized=False)
        measure = Number(1.)

        print("MEASURE: {}\n".format(measure))
        print("INTEGRAND: {}\n".format(integrand))
        result = Integrate(measure, integrand, reduced_vars)
        print("RESULT: {}\n".format(result))

    if backend == 'pyro.ops.einsum.torch_log':
        result = result.log()

    return reinterpret(result)
