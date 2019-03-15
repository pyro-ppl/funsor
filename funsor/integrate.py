from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch

import funsor.ops as ops
from funsor.distributions import Gaussian, Delta
from funsor.domains import bint
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import Finitary
from funsor.terms import Funsor, Number, Reduce, Variable, eager, reflect
from funsor.torch import Tensor


GROUND_TERMS = (Tensor, Gaussian, Delta, Number, Variable)


def _make_base_measure(arg, reduced_vars):
    if not all(isinstance(d, bint) for d in arg.inputs.values()):
        raise NotImplementedError("TODO implement continuous base measures")

    sizes = OrderedDict(set((var, dtype) for var, dtype in arg.inputs.items()))
    terms = tuple(
        Tensor(torch.ones((d.dtype,)) / float(d.dtype), OrderedDict([(var, d)]))
        for var, d in sizes.items() if var in reduced_vars
    )
    return reflect(Finitary, ops.mul, terms) if len(terms) > 1 else terms[0]


def find_constants(measure, operands):
    constants, new_operands = [], []
    for operand in operands:
        if frozenset(measure.inputs) & frozenset(operand.inputs):
            new_operands.append(operand)
        else:
            constants.append(operand)
    return tuple(constants), tuple(new_operands)


class Integrate(Funsor):

    def __init__(self, measure, integrand):
        assert isinstance(measure, Funsor)
        assert isinstance(integrand, Funsor)
        inputs = OrderedDict([(k, d) for t in (measure, integrand)
                              for k, d in t.items()])
        output = integrand.output
        super(Integrate, self).__init__(inputs, output)
        self.measure = measure
        self.integrand = integrand

    def eager_subs(self, subs):
        raise NotImplementedError("TODO implement subs")


@eager.register(Integrate, GROUND_TERMS, GROUND_TERMS)
def integrate_ground(measure, integrand):
    reduced_vars = frozenset(measure.inputs) | frozenset(integrand.inputs)
    return (measure * integrand).reduce(ops.add, reduced_vars)


@eager.register(Integrate, Funsor, Reduce)
def integrate_reduce(measure, integrand):
    if integrand.reduced_vars:
        base_measure = _make_base_measure(integrand.arg, integrand.reduced_vars)
        return Integrate(measure, Integrate(base_measure, integrand.arg))
    return Integrate(measure, integrand.arg)


@eager.register(Integrate, GROUND_TERMS, Finitary)
def integrate_finitary(measure, integrand):
    # exploit linearity of integration
    if integrand.op is ops.add:
        return Finitary(
            ops.add,
            tuple(Integrate(measure, operand) for operand in integrand.operands)
        )

    if integrand.op is ops.mul:
        # pull out terms that do not depend on the measure
        constants, new_operands = find_constants(measure, integrand.operands)

        # this term should equal Finitary(mul, constants) for probability measures
        outer = Integrate(measure, Finitary(integrand.op, constants))

        # the rest of the integral
        inner = Integrate(measure, Finitary(integrand.op, new_operands))
        return outer * inner

    return None


@eager.register(Integrate, Finitary, Finitary)
def integrate_finitary_finitary(measure, integrand):
    # exploit linearity of integration
    if integrand.op is ops.add:
        return Finitary(
            ops.add,
            tuple(Integrate(measure, operand) for operand in integrand.operands)
        )

    if integrand.op is ops.mul and measure.op is ops.mul:
        # topologically order the measures according to their variables
        assert len(measure.operands) > 1
        root_measure = measure.operands[0]
        if len(measure.operands) > 2:
            remaining_measure = Finitary(measure.op, measure.operands[1:])
        else:
            remaining_measure = measure.operands[1]

        # recursively apply law of iterated expectations
        return Integrate(root_measure, Integrate(remaining_measure, integrand))

    return None


##############################
# utilities for testing follow
##############################

def integrate_sum_product(sum_op, prod_op, factors, eliminate=frozenset()):
    """
    Utility for testing against einsum tests
    """
    assert sum_op is ops.add
    assert prod_op is ops.mul

    with interpretation(reflect):
        integrand = Finitary(prod_op, tuple(factors))
        measure = _make_base_measure(integrand, eliminate)

    return Integrate(measure, integrand)


def naive_integrate_einsum(eqn, *terms, **kwargs):
    assert "plates" not in kwargs

    backend = kwargs.pop('backend', 'torch')
    if backend in ('torch', 'pyro.ops.einsum.torch_log'):
        sum_op, prod_op = ops.add, ops.mul
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
    reduce_vars = input_dims - output_dims

    if backend == 'pyro.ops.einsum.torch_log':
        terms = tuple(term.exp() for term in terms)
    result = integrate_sum_product(sum_op, prod_op, terms, reduce_vars)
    if backend == 'pyro.ops.einsum.torch_log':
        result = result.log()
    return result


def naive_integrate(measure, integrand):
    intermediate = reinterpret(measure) * reinterpret(integrand)
    return intermediate.reduce(ops.add, frozenset(intermediate.inputs))
