from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict
from six.moves import reduce

from multipledispatch.variadic import Variadic

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import find_domain
from funsor.gaussian import Gaussian, sym_inverse
from funsor.ops import AssociativeOp, DISTRIBUTIVE_OPS
from funsor.terms import Binary, Funsor, Number, Reduce, Subs, Unary, eager, moment_matching, normalize
from funsor.torch import Tensor


class AnyOp(AssociativeOp):
    """Placeholder associative op that unifies with any other op"""
    pass


@AnyOp
def anyop(x, y):
    raise ValueError("should never actually evaluate this!")


class Contraction(Funsor):
    """
    Declarative representation of a finitary sum-product operation
    """
    def __init__(self, red_op, bin_op, reduced_vars, terms):
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
            assert (red_op, bin_op) in DISTRIBUTIVE_OPS

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


# @eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
# def eager_contraction_generic_naive(red_op, bin_op, reduced_vars, terms):
#     # push down leaf reductions
#     terms, reduced_vars = list(terms), frozenset(reduced_vars)
#     for i, v in enumerate(terms):
#         v_unique = reduced_vars.intersection(v.inputs) - \
#             frozenset().union(*(reduced_vars.intersection(vv.inputs) for vv in terms if vv is not v))
#         terms[i] = v.reduce(red_op, v_unique)
#         reduced_vars -= v_unique
#     # sum-product the remaining vars
#     return reduce(bin_op, tuple(terms)).reduce(red_op, reduced_vars)


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def eager_contraction_generic_recursive(red_op, bin_op, reduced_vars, terms):

    if len(terms) == 2:
        result = bin_op(*terms).reduce(red_op, reduced_vars)
        if result is not normalize(Contraction, red_op, bin_op, reduced_vars, terms):
            return result
        return None

    if len(terms) == 1:
        result = terms[0].reduce(red_op, reduced_vars)
        if result is not normalize(Contraction, red_op, bin_op, reduced_vars, terms):
            return result
        return None

    # push down leaf reductions
    terms, reduced_vars = list(terms), frozenset(reduced_vars)
    for i, v in enumerate(terms):
        unique_vars = reduced_vars.intersection(v.inputs) - \
            frozenset().union(*(reduced_vars.intersection(vv.inputs) for vv in terms if vv is not v))
        terms[i] = v.reduce(red_op, unique_vars)
        reduced_vars -= unique_vars

    # exploit associativity to recursively evaluate this contraction
    # a bit expensive, but handles interpreter-imposed directionality constraints
    terms = tuple(terms)
    for i, (lhs, rhs) in enumerate(zip(terms[0:-1], terms[1:])):
        unique_vars = reduced_vars.intersection(lhs.inputs, rhs.inputs) - \
            frozenset().union(*(reduced_vars.intersection(vv.inputs) for vv in terms[:i] + terms[i+2:]))
        result = Contraction(red_op, bin_op, unique_vars, lhs, rhs)
        if result is not normalize(Contraction, red_op, bin_op, unique_vars, (lhs, rhs)):  # did we make progress?
            # pick the first evaluable pair
            reduced_vars -= unique_vars
            new_terms = terms[:i] + (result,) + terms[i+2:]
            return Contraction(red_op, bin_op, reduced_vars, *new_terms)

    return None


##########################################
# Normalizing Contractions
##########################################

@normalize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def normalize_generic(red_op, bin_op, reduced_vars, *terms):
    return Contraction(red_op, bin_op, reduced_vars, tuple(terms))


@normalize.register(Contraction, AnyOp, AnyOp, frozenset, Funsor)
def normalize_trivial(red_op, bin_op, reduced_vars, term):
    assert not reduced_vars
    return term


@normalize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def normalize_contraction(red_op, bin_op, reduced_vars, terms):

    if not reduced_vars and red_op is not anyop:
        return Contraction(anyop, bin_op, reduced_vars, *terms)

    if len(terms) == 1 and bin_op is not anyop:
        return Contraction(red_op, anyop, reduced_vars, *terms)

    if red_op is anyop and bin_op is anyop:
        return terms[0]

    if red_op is bin_op:
        new_terms = tuple(v.reduce(red_op, reduced_vars) for v in terms)
        return Contraction(red_op, bin_op, frozenset(), *new_terms)

    for i, v in enumerate(terms):

        if not isinstance(v, Contraction):
            continue

        if v.red_op is anyop and (v.bin_op, bin_op) in DISTRIBUTIVE_OPS:
            # a * e * (b + c + d) -> (a * e * b) + (a * e * c) + (a * e * d)
            new_terms = tuple(
                Contraction(v.red_op, bin_op, v.reduced_vars, *(terms[:i] + (vt,) + terms[i+1:]))
                for vt in v.terms)
            return Contraction(red_op, v.bin_op, reduced_vars, *new_terms)

        if (v.red_op, bin_op) in DISTRIBUTIVE_OPS:
            new_terms = terms[:i] + (Contraction(v.red_op, v.bin_op, frozenset(), *v.terms),) + terms[i+1:]
            return Contraction(v.red_op, bin_op, v.reduced_vars, *new_terms).reduce(red_op, reduced_vars)

        if v.red_op in (red_op, anyop) and bin_op in (v.bin_op, anyop):
            red_op = v.red_op if red_op is anyop else red_op
            bin_op = v.bin_op if bin_op is anyop else bin_op
            new_terms = terms[:i] + v.terms + terms[i+1:]
            return Contraction(red_op, bin_op, reduced_vars | v.reduced_vars, *new_terms)

    # nothing more to do, reflect
    return None


#########################################
# Creating Contractions from other terms
#########################################

@normalize.register(Binary, AssociativeOp, Funsor, Funsor)
def binary_to_contract(op, lhs, rhs):
    return Contraction(anyop, op, frozenset(), lhs, rhs)


@normalize.register(Reduce, AssociativeOp, Funsor, frozenset)
def reduce_funsor(op, arg, reduced_vars):
    return Contraction(op, anyop, reduced_vars, arg)


#######################################################################
# Distributing Unary transformations (Subs, log, exp, neg, reciprocal)
#######################################################################

@normalize.register(Subs, Funsor, tuple)
def do_fresh_subs(arg, subs):
    if all(name in arg.fresh for name, sub in subs):
        return arg.eager_subs(subs)
    return None


@normalize.register(Subs, Contraction, tuple)
def distribute_subs_contraction(arg, subs):
    new_terms = tuple(Subs(v, tuple((name, sub) for name, sub in subs if name in v.inputs))
                      if any(name in v.inputs for name, sub in subs)
                      else v
                      for v in arg.terms)
    return Contraction(arg.red_op, arg.bin_op, arg.reduced_vars, *new_terms)


@normalize.register(Subs, Subs, tuple)
def normalize_fuse_subs(arg, subs):
    # a(b)(c) -> a(b(c), c)
    new_subs = subs + tuple((k, Subs(v, subs)) for k, v in arg.subs)
    return Subs(arg.arg, new_subs)


@normalize.register(Binary, ops.SubOp, Funsor, Funsor)
def binary_subtract(op, lhs, rhs):
    return lhs + -rhs


@normalize.register(Unary, ops.NegOp, Contraction)
def unary_contract(op, arg):
    if arg.bin_op is ops.add and arg.red_op is anyop:
        return Contraction(arg.red_op, arg.bin_op, arg.reduced_vars, *(-t for t in arg.terms))
    raise NotImplementedError("TODO")


@normalize.register(Unary, ops.ReciprocalOp, Contraction)
def unary_contract(op, arg):
    raise NotImplementedError("TODO")


@normalize.register(Unary, ops.TransformOp, Contraction)
def unary_transform(op, arg):
    if op is ops.log:
        raise NotImplementedError("TODO")
    elif op is ops.exp:
        raise NotImplementedError("TODO")

    return None


#################################
# patterns for joint integration
#################################

# @eager.register(Contraction, AssociativeOp, ops.AddOp, frozenset, Delta, Delta)
# def eager_contract_deltas(red_op, bin_op, reduced_vars, lhs, rhs):  # TODO only need Binary
#
#     if red_op not in (ops.logaddexp, ops.add, anyop):
#         return None
#
#     # substitute in all shared deltas
#     assert lhs.name not in rhs.inputs or rhs.name not in rhs.inputs
#     lhs = lhs(**{rhs.name: rhs.point}) if rhs.name in lhs.inputs else lhs
#     rhs = rhs(**{lhs.name: lhs.point}) if lhs.name in rhs.inputs else rhs
#
#     # TODO handle extra inputs of dropped deltas
#     if reduced_vars or lhs is not lhs or rhs is not rhs:
#         return bin_op(lhs, rhs).reduce(red_op, reduced_vars)
#
#     return None


@eager.register(Contraction, AssociativeOp, ops.AddOp, frozenset, Variadic[(Delta, Gaussian, Number, Tensor)])
def eager_contract_joint(red_op, bin_op, reduced_vars, *terms):

    if not any(isinstance(t, Delta) for t in terms) or red_op not in (ops.logaddexp, ops.add, anyop):
        return None

    # group terms
    deltas = reduce(bin_op, (t for t in terms if isinstance(t, Delta)))
    other = reduce(bin_op, (t for t in terms if not isinstance(t, Delta)))

    # sum out shared Deltas
    if red_op is ops.logaddexp:
        delta_subs = OrderedDict((t.name, t.point) for t in deltas.terms
                                 if isinstance(t, Delta) and t.name in reduced_vars
                                 and t.name in other.inputs)

        if delta_subs:
            other = other(**delta_subs)

    new_terms = (deltas, other)
    if reduced_vars or len(terms) > len(new_terms) or any(v is not t for t, v in zip(new_terms, terms)):
        return Contraction(red_op, bin_op, reduced_vars, *new_terms)

    # terminate recursion
    return None


@moment_matching.register(Contraction, AssociativeOp, ops.AddOp, frozenset, (Number, Tensor), Gaussian)
def moment_matching_contract_joint(red_op, bin_op, reduced_vars, discrete, gaussian):

    if red_op is not ops.logaddexp:
        return None

    approx_vars = frozenset(k for k in reduced_vars if gaussian.inputs.get(k, 'real') != 'real')
    exact_vars = reduced_vars - approx_vars

    if exact_vars and approx_vars:
        return Contraction(red_op, bin_op, exact_vars, discrete, gaussian).reduce(red_op, approx_vars)

    if approx_vars and not exact_vars:
        new_discrete = discrete.reduce(ops.logaddexp, approx_vars.intersection(discrete.inputs))
        num_elements = reduce(ops.mul, [
            gaussian.inputs[k].num_elements for k in approx_vars.difference(discrete.inputs)], 1)
        if num_elements != 1:
            new_discrete -= math.log(num_elements)

        int_inputs = OrderedDict((k, d) for k, d in gaussian.inputs.items() if d.dtype != 'real')
        probs = (discrete - new_discrete).exp()
        old_loc = Tensor(gaussian.loc, int_inputs)
        new_loc = (probs * old_loc).reduce(ops.add, approx_vars)
        old_cov = Tensor(sym_inverse(gaussian.precision), int_inputs)
        diff = old_loc - new_loc
        outers = Tensor(diff.data.unsqueeze(-1) * diff.data.unsqueeze(-2), diff.inputs)
        new_cov = ((probs * old_cov).reduce(ops.add, approx_vars) +
                   (probs * outers).reduce(ops.add, approx_vars))
        new_precision = Tensor(sym_inverse(new_cov.data), new_cov.inputs)
        new_inputs = new_loc.inputs.copy()
        new_inputs.update((k, d) for k, d in gaussian.inputs.items() if d.dtype == 'real')
        new_gaussian = Gaussian(new_loc.data, new_precision.data, new_inputs)
        return new_discrete + new_gaussian

    return None
