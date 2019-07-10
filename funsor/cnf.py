from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from six.moves import reduce

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import find_domain
from funsor.gaussian import Gaussian
from funsor.ops import AssociativeOp, DISTRIBUTIVE_OPS, NegOp, SubOp, TransformOp
from funsor.terms import Binary, Funsor, Number, Reduce, Subs, Unary, Variable, eager, normalize, reflect
from funsor.torch import Tensor


class AnyOp(AssociativeOp):
    pass


@AnyOp
def anyop(x, y):
    raise ValueError("should never actually evaluate this!")


class Contraction(Funsor):
    """
    Declarative representation of a finitary sum-product operation
    """
    def __init__(self, red_op, bin_op, terms, reduced_vars):
        assert isinstance(red_op, AssociativeOp)
        assert isinstance(bin_op, AssociativeOp)
        assert all(isinstance(v, Funsor) for v in terms)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, str) for v in reduced_vars)

        assert isinstance(terms, tuple) and len(terms) > 0
        assert not (isinstance(red_op, AnyOp) and isinstance(bin_op, AnyOp))
        if isinstance(red_op, AnyOp):
            assert not reduced_vars
        elif isinstance(bin_op, AnyOp):
            assert len(terms) == 1
        else:
            assert reduced_vars and len(terms) > 1

        inputs = OrderedDict()
        for v in terms:
            inputs.update(v.inputs)
        if bin_op is anyop:
            output = terms[0].output
        else:
            output = reduce(lambda lhs, rhs: find_domain(bin_op, lhs, rhs),
                            [v.output for v in reversed(terms)])
        fresh = frozenset()
        bound = reduced_vars
        super(Contraction, self).__init__(inputs, output, fresh, bound)
        self.red_op = red_op
        self.bin_op = bin_op
        self.terms = terms
        self.reduced_vars = reduced_vars


@eager.register(Contraction, AssociativeOp, AssociativeOp, tuple, frozenset)
def eager_contraction(red_op, bin_op, terms, reduced_vars):
    return reduce(bin_op, terms).reduce(red_op, reduced_vars)


##########################################
# Normalizing Contractions
##########################################

@normalize.register(Contraction, AssociativeOp, AssociativeOp, tuple, frozenset)
def normalize_contraction(red_op, bin_op, terms, reduced_vars):

    if not reduced_vars and not isinstance(red_op, AnyOp):
        return Contraction(anyop, bin_op, terms, reduced_vars)

    if len(terms) == 1 and not isinstance(bin_op, AnyOp):
        return Contraction(red_op, anyop, terms, reduced_vars)

    if red_op is bin_op:
        new_terms = tuple(v.reduce(red_op, reduced_vars) for v in terms)
        return Contraction(red_op, bin_op, new_terms, frozenset())

    for i, v in enumerate(terms):
        if not isinstance(v, Contraction):
            continue

        if (v.red_op, bin_op) in DISTRIBUTIVE_OPS:
            new_terms = terms[:i] + (Contraction(anyop, v.bin_op, v.terms, frozenset()),) + terms[i+1:]
            return Contraction(v.red_op, bin_op, new_terms, v.reduced_vars).reduce(red_op, reduced_vars)

        if isinstance(v.red_op, AnyOp) and (v.bin_op, bin_op) in DISTRIBUTIVE_OPS:
            # a * e * (b + c + d) -> (a * e * b) + (a * e * c) + (a * e * d)
            new_terms = tuple(
                Contraction(v.red_op, bin_op, terms[:i] + (vt,) + terms[i+1:], v.reduced_vars)
                for vt in v.terms)
            return Contraction(red_op, v.bin_op, new_terms, reduced_vars)

        if isinstance(v.red_op, AnyOp) and bin_op is v.bin_op:
            # a + (b + c) -> (a + b + c)
            new_terms = terms[:i] + v.terms + terms[i+1:]
            return Contraction(red_op, bin_op, new_terms, reduced_vars)

        if red_op is v.red_op and (isinstance(bin_op, AnyOp) or bin_op is v.bin_op):
            new_terms = terms[:i] + v.terms + terms[i+1:]
            return Contraction(red_op, bin_op, new_terms, reduced_vars | v.reduced_vars)

        # raise NotImplementedError("this case is not handled for some reason")

    # nothing to do, reflect
    return None


#########################################
# Creating Contractions from other terms
#########################################

@normalize.register(Binary, AssociativeOp, Funsor, Funsor)
def binary_to_contract(op, lhs, rhs):
    return Contraction(anyop, op, (lhs, rhs), frozenset())


@normalize.register(Reduce, AssociativeOp, Funsor, frozenset)
def reduce_funsor(op, arg, reduced_vars):
    return Contraction(op, anyop, (arg,), reduced_vars)


#######################################################################
# Distributing Unary transformations (Subs, log, exp, neg, reciprocal)
#######################################################################

@normalize.register(Subs, Contraction, tuple)
def distribute_subs_contraction(arg, subs):
    new_terms = tuple(Subs(v, subs) if any(k in v.inputs for k, s in subs) else v
                      for v in arg.terms)
    return Contraction(arg.red_op, arg.bin_op, new_terms, arg.reduced_vars)


@normalize.register(Binary, SubOp, Funsor, Funsor)
def binary_subtract(op, lhs, rhs):
    return lhs + -rhs


@normalize.register(Unary, NegOp, Contraction)
def unary_contract(op, arg):
    raise NotImplementedError("TODO")


@normalize.register(Unary, TransformOp, Contraction)
def unary_transform(op, arg):
    if op is ops.log:
        raise NotImplementedError("TODO")
    elif op is ops.exp:
        raise NotImplementedError("TODO")

    return None


#########################
# Joint density
#########################

class CJoint(Contraction):
    """
    Contraction that guarantees all deltas are substituted
    """
    def __init__(self, red_op, terms, reduced_vars):
        assert all(isinstance(v, (Delta, Gaussian, Tensor, Number)) for v in terms)
        super(CJoint, self).__init__(red_op, ops.add, terms, reduced_vars)


@normalize.register(CJoint, AssociativeOp, tuple, frozenset)  # TODO
def normalize_cjoint(red_op, terms, reduced_vars):
    v = normalize_contraction(red_op, ops.add, terms, reduced_vars)
    try:
        return reflect(CJoint, v.red_op, v.terms, v.reduced_vars) if isinstance(v, Contraction) else v
    except AssertionError:
        return v


#########################
# Affine contraction
#########################

class CAffine(Contraction):
    """
    Contraction that guarantees it is a multilinear function of its variable terms
    """
    def __init__(self, red_op, bin_op, terms, reduced_vars):
        assert all(isinstance(v, (Number, Tensor, Variable)) for v in terms)
        assert any(isinstance(v, Variable) for v in terms)
        var_terms = tuple(v for v in terms if isinstance(v, Variable))
        assert var_terms == tuple(frozenset(var_terms))  # each var can appear once
        super(CAffine, self).__init__(red_op, bin_op, terms, reduced_vars)


@normalize.register(CAffine, AssociativeOp, AssociativeOp, tuple, frozenset)
def normalize_caffine(red_op, bin_op, terms, reduced_vars):
    v = normalize_contraction(red_op, bin_op, terms, reduced_vars)
    try:
        return reflect(CAffine, v.red_op, v.bin_op, v.terms, v.reduced_vars) if isinstance(v, Contraction) else v
    except AssertionError:
        return v


#######################
# Finitary replacement
#######################

class CFinitary(Contraction):
    """
    Contraction that does not reduce (was used in optimizer as IR)
    """
    def __init__(self, bin_op, terms):
        assert isinstance(bin_op, AssociativeOp)  # XXX CommutativeOp?
        super(CFinitary, self).__init__(anyop, bin_op, terms, frozenset())


@normalize.register(CFinitary, AssociativeOp, tuple)
def normalize_finitary(bin_op, terms):
    v = normalize_contraction(anyop, bin_op, terms, frozenset())
    try:
        return reflect(CFinitary, v.bin_op, v.terms) if isinstance(v, Contraction) else v
    except AssertionError:
        return v
