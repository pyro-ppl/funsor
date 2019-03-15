from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch

import funsor.ops as ops
from funsor.distributions import Gaussian, Delta
from funsor.interpreter import interpretation
from funsor.optimizer import Finitary
from funsor.terms import Funsor, Number, Variable, reflect
from funsor.torch import Tensor


GROUND_TERMS = (Tensor, Gaussian, Delta, Number, Variable)


def find_constants(measure, operands):
    constants, new_operands = [], []
    for operand in operands:
        if frozenset(measure.inputs) & frozenset(operand.inputs):
            new_operands.append(operand)
        else:
            constants.append(operand)
    return tuple(constants), tuple(new_operands)


def integrate(measure, integrand):
    assert isinstance(measure, (Finitary,) + GROUND_TERMS)
    assert isinstance(integrand, (Finitary,) + GROUND_TERMS)

    if isinstance(integrand, GROUND_TERMS):
        # base case of the recursion
        reduced_vars = frozenset(measure.inputs) | frozenset(integrand.inputs)
        return (measure * integrand).reduce(ops.add, reduced_vars)
        # return eager_integrate(measure, integrand)

    # exploit linearity of integration
    elif isinstance(integrand, Finitary) and isinstance(integrand.op, ops.AddOp):
        return Finitary(
            ops.add,
            tuple(integrate(measure, operand) for operand in integrand.operands)
        )

    elif isinstance(integrand, Finitary) and integrand.op is ops.mul:

        # pull out terms that do not depend on the measure
        constants, new_operands = find_constants(measure, integrand.operands)
        new_integrand = Finitary(integrand.op, new_operands)

        if isinstance(measure, Finitary) and measure.op is ops.mul:
            # topologically order the measures according to their variables
            assert len(measure.operands) > 1
            root_measure = measure.operands[0]
            if len(measure.operands) > 2:
                remaining_measure = Finitary(measure.op, measure.operands[1:])
            else:
                remaining_measure = measure.operands[1]

            # recursively apply law of iterated expectations
            inner = integrate(root_measure, integrate(remaining_measure, new_integrand))
        else:
            inner = integrate(measure, new_integrand)

        return Finitary(ops.mul, constants + (inner,))
    else:
        raise NotImplementedError("TODO implement any other cases")


def integrate_sum_product(sum_op, prod_op, factors, eliminate=frozenset()):
    """
    Utility for testing against einsum tests
    """
    assert sum_op is ops.add
    assert prod_op is ops.mul

    with interpretation(reflect):
        sizes = OrderedDict(set((var, dtype) for factor in factors
                                for var, dtype in factor.inputs.items()))
        var_tensors = tuple(
            Tensor(torch.ones((size.dtype,)) / float(size.dtype),
                   OrderedDict([(var, size)]))
            for var, size in sizes.items() if var in eliminate
        )
        measure = Finitary(prod_op, var_tensors) if len(var_tensors) > 1 else var_tensors[0]
        integrand = Finitary(prod_op, tuple(factors))

    return integrate(measure, integrand)


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
