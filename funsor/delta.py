from collections import OrderedDict
from functools import reduce

import funsor.ops as ops
import funsor.terms
from funsor.domains import Domain, reals
from funsor.integrate import Integrate
from funsor.interpreter import debug_logged
from funsor.ops import AddOp, SubOp, TransformOp
from funsor.registry import KeyedRegistry
from funsor.terms import (
    Align,
    Binary,
    Funsor,
    FunsorMeta,
    Independent,
    Lambda,
    Number,
    Reduce,
    Subs,
    Unary,
    Variable,
    eager,
    to_funsor
)


def Delta(name, point, log_density=0):
    """Syntactic sugar for MultiDelta"""
    point, log_density = to_funsor(point), to_funsor(log_density)
    return MultiDelta(((name, (point, log_density)),))


def solve(expr, value):
    """
    Tries to solve for free inputs of an ``expr`` such that ``expr == value``,
    and computes the log-abs-det-Jacobian of the resulting substitution.

    :param Funsor expr: An expression with a free variable.
    :param Funsor value: A target value.
    :return: A tuple ``(name, point, log_abs_det_jacobian)``
    :rtype: tuple
    :raises: ValueError
    """
    assert isinstance(expr, Funsor)
    assert isinstance(value, Funsor)
    result = solve.dispatch(type(expr), *(expr._ast_values + (value,)))
    if result is None:
        raise ValueError("Cannot substitute into a Delta: {}".format(value))
    return result


_solve = KeyedRegistry(lambda *args: None)
solve.dispatch = _solve.__call__
solve.register = _solve.register


@solve.register(Variable, str, Domain, Funsor)
@debug_logged
def solve_variable(name, output, y):
    assert y.output == output
    point = y
    log_density = Number(0)
    return name, point, log_density


@solve.register(Unary, TransformOp, Funsor, Funsor)
@debug_logged
def solve_unary(op, arg, y):
    x = op.inv(y)
    name, point, log_density = solve(arg, x)
    log_density += op.log_abs_det_jacobian(x, y)
    return name, point, log_density


class MultiDeltaMeta(FunsorMeta):
    """
    Wrapper to fill in defaults.
    """
    def __call__(cls, terms):
        terms = tuple(terms.items()) if isinstance(terms, OrderedDict) else terms
        terms = tuple((name, (to_funsor(point), to_funsor(log_density)))
                      for name, (point, log_density) in terms)
        return super(MultiDeltaMeta, cls).__call__(terms)


class MultiDelta(Funsor, metaclass=MultiDeltaMeta):
    """
    Normalized delta distribution binding multiple variables.
    """
    def __init__(self, terms):
        assert isinstance(terms, tuple) and len(terms) > 0
        inputs = OrderedDict()
        for term in terms:
            assert len(term) == 2
            name, (point, log_density) = term
            assert isinstance(name, str)
            assert isinstance(point, Funsor)
            assert isinstance(log_density, Funsor)
            assert log_density.output == reals()
            assert name not in inputs
            inputs.update({name: point.output})
            inputs.update(point.inputs)
            inputs.update(log_density.inputs)

        output = reals()
        fresh = frozenset(name for name, (point, log_density) in terms)
        bound = frozenset()
        super(MultiDelta, self).__init__(inputs, output, fresh, bound)
        self.terms = terms

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.fresh for name in names)
        if not names or names == tuple(n for n, (p, ld) in self.terms):
            return self

        new_terms = sorted(self.terms, key=lambda t: names.index(t[0]))
        return MultiDelta(new_terms)

    def eager_subs(self, subs):
        terms = OrderedDict(self.terms)
        new_terms = terms.copy()
        constant = Number(0)
        for name, value in subs:
            if isinstance(value, (str, Variable)):
                value = to_funsor(value, self.output)
                new_terms[value.name] = new_terms.pop(name)
                continue

            if not any(d.dtype == 'real' for side in (value, terms[name][0])
                       for d in side.inputs.values()):
                point, log_density = new_terms.pop(name)
                constant += (value == point).all().log() + log_density
                continue

            # Try to invert the substitution.
            soln = solve(value, terms[name][0])
            if soln is None:
                return None  # lazily substitute
            new_name, point, log_density = soln
            log_density += new_terms.pop(name)[1]
            new_terms[new_name] = (point, log_density)

        return MultiDelta(new_terms) + constant if new_terms else constant

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if any(name in reduced_vars for name, (point, ld) in self.terms):
                result_terms = tuple((name, (point, ld))
                                     for name, (point, ld) in self.terms
                                     if name not in reduced_vars)
                result = MultiDelta(result_terms) if result_terms else Number(0)
                result += ops.log(
                    reduce(ops.mul, [self.inputs[v].dtype for v in reduced_vars - self.fresh], Number(1)))
                return result  # Deltas are normalized.
            return self

        # TODO Implement ops.add to simulate .to_event().

        return None  # defer to default implementation

    def unscaled_sample(self, sampled_vars, sample_inputs):
        return self


eager.register(Binary, AddOp, MultiDelta, Reduce)(
    funsor.terms.eager_distribute_other_reduce)
eager.register(Binary, AddOp, Reduce, MultiDelta)(
    funsor.terms.eager_distribute_reduce_other)


@eager.register(Binary, AddOp, MultiDelta, MultiDelta)
def eager_add_multidelta(op, lhs, rhs):
    if lhs.fresh.intersection(rhs.inputs):
        return eager_add_delta_funsor(op, lhs, rhs)

    if rhs.fresh.intersection(lhs.inputs):
        return eager_add_funsor_delta(op, lhs, rhs)

    return MultiDelta(lhs.terms + rhs.terms)


@eager.register(Binary, (AddOp, SubOp), MultiDelta, (Funsor, Align))
def eager_add_delta_funsor(op, lhs, rhs):
    if lhs.fresh.intersection(rhs.inputs):
        rhs = rhs(**{name: point for name, (point, ld) in lhs.terms if name in rhs.inputs})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, AddOp, (Funsor, Align), MultiDelta)
def eager_add_funsor_delta(op, lhs, rhs):
    return eager_add_delta_funsor(op, rhs, lhs)  # XXX is this pattern redundant? it should be


@eager.register(Integrate, MultiDelta, Funsor, frozenset)
def eager_integrate(delta, integrand, reduced_vars):
    assert reduced_vars & delta.fresh
    subs = tuple((name, point) for name, (point, log_density) in delta.terms
                 if name in reduced_vars)
    integrand = Subs(integrand, subs)
    log_measure = Subs(delta, subs)
    reduced_vars -= delta.fresh
    return Integrate(log_measure, integrand, reduced_vars)


@eager.register(Independent, MultiDelta, str, str)
def eager_independent(delta, reals_var, bint_var):
    # if delta.name == reals_var or delta.name.startswith(reals_var + "__BOUND"):
    #     i = Variable(bint_var, delta.inputs[bint_var])
    #     point = Lambda(i, delta.point)
    #     if bint_var in delta.log_density.inputs:
    #         log_density = delta.log_density.reduce(ops.add, bint_var)
    #     else:
    #         log_density = delta.log_density * delta.inputs[bint_var].dtype
    #     return Delta(reals_var, point, log_density)

    raise NotImplementedError("TODO")
    # return None  # defer to default implementation


__all__ = [
    'Delta',
    'MultiDelta',
    'solve',
]
