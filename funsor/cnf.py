from collections import OrderedDict
from functools import reduce
from typing import Tuple, Union

import opt_einsum
import torch
from multipledispatch.variadic import Variadic

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import bint, find_domain
from funsor.gaussian import Gaussian
from funsor.interpreter import gensym, recursion_reinterpret
from funsor.ops import DISTRIBUTIVE_OPS, AssociativeOp, NullOp, nullop
from funsor.terms import (
    Align,
    Binary,
    Funsor,
    Lambda,
    Number,
    Reduce,
    Subs,
    Unary,
    Variable,
    eager,
    normalize,
    to_funsor
)
from funsor.torch import Tensor
from funsor.util import quote


class Contraction(Funsor):
    """
    Declarative representation of a finitary sum-product operation

    :ivar bool is_affine: Whether this contraction is affine.
    """
    def __init__(self, red_op, bin_op, reduced_vars, terms):
        terms = (terms,) if isinstance(terms, Funsor) else terms
        assert isinstance(red_op, AssociativeOp)
        assert isinstance(bin_op, AssociativeOp)
        assert all(isinstance(v, Funsor) for v in terms)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, str) for v in reduced_vars)
        assert isinstance(terms, tuple) and len(terms) > 0

        assert not (isinstance(red_op, NullOp) and isinstance(bin_op, NullOp))
        if isinstance(red_op, NullOp):
            assert not reduced_vars
        elif isinstance(bin_op, NullOp):
            assert len(terms) == 1
        else:
            assert reduced_vars and len(terms) > 1
            assert (red_op, bin_op) in DISTRIBUTIVE_OPS

        inputs = OrderedDict()
        for v in terms:
            inputs.update((k, d) for k, d in v.inputs.items() if k not in reduced_vars)

        if bin_op is nullop:
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
        self.is_affine = self._is_affine()

    def _is_affine(self):
        for t in self.terms:
            if not isinstance(t, (Number, Tensor, Variable, Contraction)):
                return False
            if isinstance(t, Contraction):
                if not ((t.bin_op, self.bin_op) in DISTRIBUTIVE_OPS or (self.bin_op, t.bin_op) in DISTRIBUTIVE_OPS) \
                        and t.is_affine:
                    return False

        if self.bin_op is ops.add and self.red_op is not nullop:
            return sum(1 for k, v in self.inputs.items() if v.dtype == 'real') == \
                sum(sum(1 for k, v in t.inputs.items() if v.dtype == 'real') for t in self.terms)
        return True

    def unscaled_sample(self, sampled_vars, sample_inputs):
        if self.red_op in (ops.logaddexp, nullop) and self.bin_op in (ops.logaddexp, ops.add, nullop):
            new_terms = tuple(v.unscaled_sample(sampled_vars.intersection(v.inputs), sample_inputs) for v in self.terms)
            return Contraction(self.red_op, self.bin_op, self.reduced_vars, *new_terms)
        raise TypeError("Cannot sample through ops ({}, {})".format(self.red_op, self.bin_op))

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        new_terms = tuple(t.align(tuple(n for n in names if n in t.inputs)) for t in self.terms)
        result = Contraction(self.red_op, self.bin_op, self.reduced_vars, *new_terms)
        if not names == tuple(result.inputs):
            return Align(result, names)  # raise NotImplementedError("TODO align all terms")
        return result

    def _alpha_convert(self, alpha_subs):
        reduced_vars = frozenset(alpha_subs.get(k, k) for k in self.reduced_vars)
        bound_types = {}
        for term in self.terms:
            bound_types.update({k: term.inputs[k] for k in self.bound.intersection(term.inputs)})
        alpha_subs = {k: to_funsor(v, bound_types[k]) for k, v in alpha_subs.items()}
        red_op, bin_op, _, terms = super()._alpha_convert(alpha_subs)
        return red_op, bin_op, reduced_vars, terms

    def extract_affine(self):
        """
        Returns a pair ``(const, coeffs)`` where const is a funsor with no real
        inputs and ``coeffs`` is an OrderedDict mapping input name to a
        ``(coefficient, eqn)`` pair in einsum form, i.e. satisfying::

            x = Contraction(...)
            assert x.is_affine
            const, coeffs = x.extract_affine()
            y = sum(Einsum(eqn, (coeff, Variable(var, coeff.output)))
                    for var, (coeff, eqn) in coeffs.items())
            assert_close(y, x)

        This only works for affine funsors. Check with :ivar:`.is_affine`
        """
        assert self.is_affine
        real_inputs = OrderedDict((k, v) for k, v in self.inputs.items() if v.dtype == 'real')
        coeffs = OrderedDict()
        zeros = {k: Tensor(torch.zeros(v.shape)) for k, v in real_inputs.items()}
        const = self(**zeros)
        name = gensym('probe')
        for k, v in real_inputs.items():
            dim = v.num_elements
            var = Variable(name, bint(dim))
            subs = zeros.copy()
            subs[k] = Tensor(torch.eye(dim).reshape((dim,) + v.shape))[var]
            coeff = Lambda(var, self(**subs) - const).reshape(v.shape + const.shape)
            inputs1 = ''.join(map(opt_einsum.get_symbol, range(len(coeff.shape))))
            inputs2 = inputs1[:len(v.shape)]
            output = inputs1[len(v.shape):]
            eqn = f"...{inputs1},...{inputs2}->...{output}"
            coeffs[k] = coeff, eqn
        return const, coeffs


@quote.register(Contraction)
def _(arg, indent, out):
    line = f"{type(arg).__name__}({repr(arg.red_op)}, {repr(arg.bin_op)},"
    out.append((indent, line))
    quote.inplace(arg.reduced_vars, indent + 1, out)
    i, line = out[-1]
    out[-1] = i, line + ","
    quote.inplace(arg.terms, indent + 1, out)
    i, line = out[-1]
    out[-1] = i, line + ")"


@recursion_reinterpret.register(Contraction)
def recursion_reinterpret_contraction(x):
    return type(x)(*map(recursion_reinterpret, (x.red_op, x.bin_op, x.reduced_vars) + x.terms))


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def eager_contraction_generic_to_tuple(red_op, bin_op, reduced_vars, *terms):
    return eager(Contraction, red_op, bin_op, reduced_vars, tuple(terms))


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def eager_contraction_generic_recursive(red_op, bin_op, reduced_vars, terms):

    # push down leaf reductions
    terms, reduced_vars, leaf_reduced = list(terms), frozenset(reduced_vars), False
    for i, v in enumerate(terms):
        unique_vars = reduced_vars.intersection(v.inputs) - \
            frozenset().union(*(reduced_vars.intersection(vv.inputs) for vv in terms if vv is not v))
        if unique_vars:
            result = v.reduce(red_op, unique_vars)
            if result is not normalize(Contraction, red_op, nullop, unique_vars, (v,)):
                terms[i] = result
                reduced_vars -= unique_vars
                leaf_reduced = True

    if leaf_reduced:
        return Contraction(red_op, bin_op, reduced_vars, *terms)

    # exploit associativity to recursively evaluate this contraction
    # a bit expensive, but handles interpreter-imposed directionality constraints
    terms = tuple(terms)
    # return reduce(bin_op, terms).reduce(red_op, reduced_vars)
    # for i, (lhs, rhs) in enumerate(zip(terms[0:-1], terms[1:])):
    for i, lhs in enumerate(terms[0:-1]):
        for j_, rhs in enumerate(terms[i+1:]):
            j = i + j_ + 1
            unique_vars = reduced_vars.intersection(lhs.inputs, rhs.inputs) - \
                frozenset().union(*(reduced_vars.intersection(vv.inputs)
                                    for vv in terms[:i] + terms[i+1:j] + terms[j+1:]))
            result = Contraction(red_op, bin_op, unique_vars, lhs, rhs)
            if result is not normalize(Contraction, red_op, bin_op, unique_vars, (lhs, rhs)):  # did we make progress?
                # pick the first evaluable pair
                reduced_vars -= unique_vars
                new_terms = terms[:i] + (result,) + terms[i+1:j] + terms[j+1:]
                return Contraction(red_op, bin_op, reduced_vars, *new_terms)

    return None


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor)
def eager_contraction_to_reduce(red_op, bin_op, reduced_vars, term):
    return eager.dispatch(Reduce, red_op, term, reduced_vars)


@eager.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor, Funsor)
def eager_contraction_to_binary(red_op, bin_op, reduced_vars, lhs, rhs):

    if reduced_vars - (reduced_vars.intersection(lhs.inputs, rhs.inputs)):
        result = eager.dispatch(Contraction, red_op, bin_op, reduced_vars, (lhs, rhs))
        if result is not None:
            return result

    result = eager.dispatch(Binary, bin_op, lhs, rhs)
    if result is not None and reduced_vars:
        result = eager.dispatch(Reduce, red_op, result, reduced_vars)
    return result


##########################################
# Normalizing Contractions
##########################################

GROUND_TERMS = (Delta, Gaussian, Number, Tensor)
GaussianMixture = Contraction[Union[ops.LogAddExpOp, NullOp], ops.AddOp, frozenset,
                              Tuple[Union[Tensor, Number], Gaussian]]


@normalize.register(Contraction, AssociativeOp, ops.AddOp, frozenset, GROUND_TERMS, GROUND_TERMS)
def normalize_contraction_commutative_canonical_order(red_op, bin_op, reduced_vars, *terms):
    # when bin_op is commutative, put terms into a canonical order for pattern matching
    ordering = {Delta: 1, Number: 2, Tensor: 3, Gaussian: 4}
    new_terms = tuple(
        v for i, v in sorted(enumerate(terms),
                             key=lambda t: (ordering.get(type(t[1]).__origin__, -1), t[0]))
    )
    if any(v is not vv for v, vv in zip(terms, new_terms)):
        return Contraction(red_op, bin_op, reduced_vars, *new_terms)
    return normalize(Contraction, red_op, bin_op, reduced_vars, new_terms)


@normalize.register(Contraction, AssociativeOp, ops.AddOp, frozenset, GaussianMixture, GROUND_TERMS)
def normalize_contraction_commute_joint(red_op, bin_op, reduced_vars, mixture, other):
    return Contraction(mixture.red_op if red_op is nullop else red_op, bin_op,
                       reduced_vars | mixture.reduced_vars, *(mixture.terms + (other,)))


@normalize.register(Contraction, AssociativeOp, ops.AddOp, frozenset, GROUND_TERMS, GaussianMixture)
def normalize_contraction_commute_joint(red_op, bin_op, reduced_vars, other, mixture):
    return Contraction(mixture.red_op if red_op is nullop else red_op, bin_op,
                       reduced_vars | mixture.reduced_vars, *(mixture.terms + (other,)))


@normalize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def normalize_contraction_generic_args(red_op, bin_op, reduced_vars, *terms):
    return normalize(Contraction, red_op, bin_op, reduced_vars, tuple(terms))


@normalize.register(Contraction, NullOp, NullOp, frozenset, Funsor)
def normalize_trivial(red_op, bin_op, reduced_vars, term):
    assert not reduced_vars
    return term


@normalize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def normalize_contraction_generic_tuple(red_op, bin_op, reduced_vars, terms):

    if not reduced_vars and red_op is not nullop:
        return Contraction(nullop, bin_op, reduced_vars, *terms)

    if len(terms) == 1 and bin_op is not nullop:
        return Contraction(red_op, nullop, reduced_vars, *terms)

    if red_op is nullop and bin_op is nullop:
        return terms[0]

    if red_op is bin_op:
        new_terms = tuple(v.reduce(red_op, reduced_vars) for v in terms)
        return Contraction(red_op, bin_op, frozenset(), *new_terms)

    if bin_op in ops.UNITS and any(isinstance(t, Number) and t.data == ops.UNITS[bin_op] for t in terms):
        new_terms = tuple(t for t in terms if not (isinstance(t, Number) and t.data == ops.UNITS[bin_op]))
        if not new_terms:  # everything was a unit
            new_terms = (terms[0],)
        return Contraction(red_op, bin_op, reduced_vars, *new_terms)

    for i, v in enumerate(terms):

        if not isinstance(v, Contraction):
            continue

        # fuse operations without distributing
        if (v.red_op is nullop and bin_op is v.bin_op) or \
                (bin_op is nullop and v.red_op in (red_op, nullop)):
            red_op = v.red_op if red_op is nullop else red_op
            bin_op = v.bin_op if bin_op is nullop else bin_op
            new_terms = terms[:i] + v.terms + terms[i+1:]
            return Contraction(red_op, bin_op, reduced_vars | v.reduced_vars, *new_terms)

    # nothing more to do, reflect
    return None


#########################################
# Creating Contractions from other terms
#########################################

@normalize.register(Binary, AssociativeOp, Funsor, Funsor)
def binary_to_contract(op, lhs, rhs):
    return Contraction(nullop, op, frozenset(), lhs, rhs)


@normalize.register(Reduce, AssociativeOp, Funsor, frozenset)
def reduce_funsor(op, arg, reduced_vars):
    return Contraction(op, nullop, reduced_vars, arg)


@normalize.register(Unary, ops.NegOp, (Variable, Contraction[ops.AssociativeOp, ops.MulOp, frozenset, tuple]))
def unary_neg_variable(op, arg):
    return arg * -1


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


@normalize.register(Binary, ops.DivOp, Funsor, Funsor)
def binary_divide(op, lhs, rhs):
    return lhs * Unary(ops.reciprocal, rhs)


@normalize.register(Unary, ops.ExpOp, Unary[ops.LogOp, Funsor])
@normalize.register(Unary, ops.LogOp, Unary[ops.ExpOp, Funsor])
@normalize.register(Unary, ops.NegOp, Unary[ops.NegOp, Funsor])
@normalize.register(Unary, ops.ReciprocalOp, Unary[ops.ReciprocalOp, Funsor])
def unary_log_exp(op, arg):
    return arg.arg


@normalize.register(Unary, ops.ReciprocalOp, Contraction[NullOp, ops.MulOp, frozenset, tuple])
@normalize.register(Unary, ops.NegOp, Contraction[NullOp, ops.AddOp, frozenset, tuple])
def unary_contract(op, arg):
    return Contraction(arg.red_op, arg.bin_op, arg.reduced_vars, *(op(t) for t in arg.terms))


@normalize.register(Unary, ops.LogOp,
                    Contraction[Union[ops.AddOp, NullOp], Union[ops.MulOp, NullOp], frozenset, tuple])
def unary_transform_log(op, arg):
    new_terms = tuple(v.log() for v in arg.terms)
    return Contraction(ops.logaddexp, ops.add, arg.reduced_vars, *new_terms)


@normalize.register(Unary, ops.ExpOp,
                    Contraction[Union[ops.LogAddExpOp, NullOp], Union[ops.AddOp, NullOp], frozenset, tuple])
def unary_transform_exp(op, arg):
    new_terms = tuple(v.exp() for v in arg.terms)
    return Contraction(ops.add, ops.mul, arg.reduced_vars, *new_terms)
