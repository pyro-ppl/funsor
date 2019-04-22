from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict
from six import add_metaclass

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.integrate import Integrate
from funsor.product import Product
from funsor.rvs import DeltaRV, RandomVariable
from funsor.terms import Funsor, FunsorMeta, Subs, eager


class ExpectationMeta(FunsorMeta):

    def __call__(cls, measure, integrand, sum_names=None):
        if sum_names is None:
            if isinstance(measure, Product):
                sum_names = frozenset(measure.outputs.keys())
            else:
                raise ValueError("Must declare sum_names if measure isn't a Product")
        elif isinstance(sum_names, str):
            sum_names = frozenset([sum_names])
        return super(ExpectationMeta, cls).__call__(measure, integrand, sum_names)


@add_metaclass(ExpectationMeta)
class Expectation(Funsor):
    def __init__(self, measure, integrand, sum_names):
        assert isinstance(measure, Funsor)
        assert isinstance(integrand, Funsor)
        assert isinstance(sum_names, frozenset)
        inputs = measure.inputs.copy()
        for k, d in integrand.inputs.items():
            if k not in sum_names:
                inputs[k] = d

        super(Expectation, self).__init__(inputs, output)

        self.measure = measure
        self.integrand = integrand
        self.sum_names = sum_names

    def eager_subs(self, subs):
        raise NotImplementedError("TODO implement eager_subs")


@eager.register(Expectation, Product, Funsor, frozenset)
def eager_expectation_product(product, integrand, sum_names):
    log_measure = sum(v.log_prob(k) for k, v in product.outputs.items()
                      if k in sum_names)
    # return Integrate(log_measure, integrand, sum_names)  # TODO reuse Integrate?
    return (log_measure.exp() * integrand).reduce(ops.add, sum_names)


@eager.register(Expectation, RandomVariable, Funsor, frozenset)
def eager_expectation_rv(rv, integrand, sum_names):
    assert len(sum_names) == 1
    return Expectation(Product({sum_name: rv for name in sum_names}), integrand)


@eager.register(Expectation, DeltaRV, Funsor, frozenset)
def eager_expectation_delta(delta, integrand, sum_names):
    assert len(sum_names) == 1
    return integrand(**{sum_name: delta.v for sum_name in sum_names}) * delta.log_density.exp()
