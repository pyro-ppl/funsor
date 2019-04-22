from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict
from six import add_metaclass
from six.moves import reduce

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.contract import Contract
from funsor.domains import Domain, find_domain
from funsor.ops import AssociativeOp
from funsor.optimizer import optimize
from funsor.terms import Binary, Funsor, FunsorMeta, Reduce, Subs, eager


def normal_form(fn, outputs):
    # connected components/topological sort algorithm for computing normal form:
    # 1. order output expressions topologically
    # 2. substitute variables along topology so there are no connected components
    assert isinstance(outputs, (tuple, dict, OrderedDict))

    def _normal_form(roots, remaining):
        if not remaining:
            return roots

        for name, node in list(remaining.items()):
            if not any(inp in remaining for inp in node.inputs):
                lhss = {inp: roots[inp] for inp in node.inputs if inp in roots}
                rhs = remaining.pop(name)
                if lhss:
                    roots[name] = Subs(rhs, Product(lhss))
                else:
                    roots[name] = rhs
        return _normal_form(roots, remaining)

    # find all nodes 
    outputs = dict(outputs) if isinstance(outputs, tuple) else outputs
    if set(outputs.keys()) & set().union(*(outp.inputs.keys() for outp in outputs.values())):
        return Product(_normal_form({}, outputs))
    return fn(outputs)


def productor(fn):
    fn = interpreter.debug_logged(fn)
    return functools.partial(normal_form, fn)


class ProductMeta(FunsorMeta):

    def __call__(cls, outputs):
        assert isinstance(outputs, (dict, OrderedDict, tuple))
        if isinstance(outputs, (dict, OrderedDict)):
            outputs = tuple(outputs.items())
        return super(ProductMeta, cls).__call__(outputs)


@add_metaclass(ProductMeta)
class Product(Funsor):
    """
    Tensor product of independent terms.

    Subs(term, Product) -> Subs(term, **Product.outputs)
    Subs(Product, term) -> Product({k: Subs(v, term)...})
    """
    def __init__(self, outputs):
        assert isinstance(outputs, tuple)
        inputs = OrderedDict()
        for name, outp in outputs:
            inputs.update(outp.inputs)

        output = reduce(lambda lhs, rhs: find_domain(ops.mul, lhs, rhs),
                        [outp.output for (name, outp) in reversed(outputs)])

        super(Product, self).__init__(inputs, output)
        self.outputs = OrderedDict(outputs)

    def __repr__(self):
        terms = ", ".join("{}={}".format(name, outp) for name, outp in self.outputs.items())
        return "Product({})".format(terms)

    def combine(self, other):  # XXX rename to update? operator overload?
        assert isinstance(other, Product), "Can only combine Products"
        assert not set(self.outputs.keys()) & set(other.outputs.keys()), \
            "Outputs of both Products must be disjoint"
        return Product(tuple(self.outputs.items()) + tuple(other.outputs.items()))


@eager.register(Product, tuple)
@optimize.register(Product, tuple)
@productor
def eager_product(outputs):
    return None  # defer to default implementation after normalizing


@eager.register(Subs, Funsor, Product)
@optimize.register(Subs, Funsor, Product)
def eager_subs_funsor_product(term, subs):
    return Subs(term, tuple(subs.outputs.items()))


@eager.register(Subs, Product, tuple)
def eager_subs_prod(prod, subs):
    assert not any(name in prod.outputs for (name, sub) in subs)
    new_outputs = OrderedDict()
    for name, outp in prod.outputs:
        new_outputs[name] = Subs(outp, subs)
    return Product(new_outputs)


def combine(*products):
    assert all(isinstance(product, Product) for product in products)
    return reduce(Product.combine, products[1:], products[0])


############################################################
# Pass-through operations (applied to the individual terms)
############################################################

# @eager.register(Binary, AssociativeOp, Product, Product)
# def eager_binary_product_product(op, lhs, rhs):
#     raise NotImplementedError("TODO implement Binary")
# 
# 
# @eager.register(Binary, AssociativeOp, Funsor, Product)
# def eager_binary_product_product(op, lhs, rhs):
#     raise NotImplementedError("TODO implement Binary")
# 
# 
# @eager.register(Binary, AssociativeOp, Product, Funsor)
# def eager_binary_product_product(op, lhs, rhs):
#     raise NotImplementedError("TODO implement Binary")
# 
# 
# @eager.register(Reduce, AssociativeOp, Product, frozenset)
# def eager_reduce_product(op, arg, reduced_vars):
#     # new_lhs = Product(op, arg.prod_op,
#     #                   {v: Variable(v, arg.inputs[v]) for v in reduced_vars})
#     # return Contract(sum_op, prod_op, Product(), arg)
#     raise NotImplementedError("TODO implement Reduce")
# 
# 
# @eager.register(Independent, Product, str, str)
# def eager_independent_product(joint, reals_var, bint_var):
#     raise NotImplementedError("TODO implement Independent")
# 
# 
# @eager.register(Lambda, Product, str)
# def eager_lambda_product(joint, bint_var):
#     raise NotImplementedError("TODO implement Lambda")
