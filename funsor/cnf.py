import math
from collections import OrderedDict
from functools import reduce

import opt_einsum
from multipledispatch.variadic import Variadic

import funsor.ops as ops
from funsor.domains import find_domain
from funsor.gaussian import Gaussian, sym_inverse
from funsor.interpreter import recursion_reinterpret
from funsor.ops import AssociativeOp, DISTRIBUTIVE_OPS
from funsor.terms import Binary, Funsor, Number, Reduce, Subs, Unary, Variable, \
    eager, moment_matching, normalize
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
        terms = (terms,) if isinstance(terms, Funsor) else terms
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
            inputs.update((k, d) for k, d in v.inputs.items() if k not in reduced_vars)

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
        self.is_affine = self._is_affine()

    def _is_affine(self):
        for t in self.terms:
            if not isinstance(t, (Number, Tensor, Variable, Contraction)):
                return False
            if isinstance(t, Contraction):
                if not (self.bin_op, t.bin_op) in DISTRIBUTIVE_OPS and t.is_affine:
                    return False

        if self.bin_op is ops.add and self.red_op is not anyop:
            return sum(1 for k, v in self.inputs.items() if v.dtype == 'real') == \
                sum(sum(1 for k, v in t.inputs.items() if v.dtype == 'real') for t in self.terms)
        return True


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
            leaf_reduced = True
            terms[i] = v.reduce(red_op, unique_vars)
            reduced_vars -= unique_vars

    if leaf_reduced:
        return Contraction(red_op, bin_op, reduced_vars, *terms)

    # exploit associativity to recursively evaluate this contraction
    # a bit expensive, but handles interpreter-imposed directionality constraints
    terms = tuple(terms)
    # return reduce(bin_op, terms).reduce(red_op, reduced_vars)
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
    if result is not None:
        result = eager.dispatch(Reduce, red_op, result, reduced_vars)
    return result


##########################################
# Normalizing Contractions
##########################################

@normalize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])
def normalize_contraction_generic_args(red_op, bin_op, reduced_vars, *terms):
    return normalize(Contraction, red_op, bin_op, reduced_vars, tuple(terms))


@normalize.register(Contraction, AnyOp, AnyOp, frozenset, Funsor)
def normalize_trivial(red_op, bin_op, reduced_vars, term):
    assert not reduced_vars
    return term


@normalize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def normalize_contraction_generic_tuple(red_op, bin_op, reduced_vars, terms):

    if not reduced_vars and red_op is not anyop:
        return Contraction(anyop, bin_op, reduced_vars, *terms)

    if len(terms) == 1 and bin_op is not anyop:
        return Contraction(red_op, anyop, reduced_vars, *terms)

    if red_op is anyop and bin_op is anyop:
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


@normalize.register(Binary, ops.DivOp, Funsor, Funsor)
def binary_divide(op, lhs, rhs):
    return lhs * Unary(ops.reciprocal, rhs)


@normalize.register(Unary, ops.NegOp, Contraction)
def unary_contract(op, arg):
    if arg.bin_op is ops.add and arg.red_op is anyop:
        return Contraction(arg.red_op, arg.bin_op, arg.reduced_vars, *(op(t) for t in arg.terms))
    if arg.bin_op is ops.mul:
        return arg * Number(-1.)
    return None


@normalize.register(Unary, ops.NegOp, Variable)
def unary_neg_variable(op, arg):
    return arg * Number(-1.)


@normalize.register(Unary, ops.ReciprocalOp, Contraction)
def unary_contract(op, arg):
    if arg.bin_op is ops.mul and arg.red_op is anyop:
        return Contraction(arg.red_op, arg.bin_op, arg.reduced_vars, *(op(t) for t in arg.terms))
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

@eager.register(Contraction, AssociativeOp, (ops.AddOp, AssociativeOp), frozenset, Tensor, Tensor)
def eager_contract(sum_op, prod_op, reduced_vars, lhs, rhs):
    if (sum_op, prod_op) == (ops.add, ops.mul):
        backend = "torch"
    elif (sum_op, prod_op) == (ops.logaddexp, ops.add):
        backend = "pyro.ops.einsum.torch_log"
    else:
        return prod_op(lhs, rhs).reduce(sum_op, reduced_vars)

    inputs = OrderedDict((k, d) for t in (lhs, rhs)
                         for k, d in t.inputs.items() if k not in reduced_vars)

    data = opt_einsum.contract(lhs.data, list(lhs.inputs),
                               rhs.data, list(rhs.inputs),
                               list(inputs), backend=backend)
    dtype = find_domain(prod_op, lhs.output, rhs.output).dtype
    return Tensor(data, inputs, dtype)


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
