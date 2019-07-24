from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from six import add_metaclass
from six.moves import reduce

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


class DeltaMeta(FunsorMeta):
    """
    Wrapper to fill in defaults.
    """
    def __call__(cls, name, point, log_density=0):
        point = to_funsor(point)
        log_density = to_funsor(log_density)
        return super(DeltaMeta, cls).__call__(name, point, log_density)


@add_metaclass(DeltaMeta)
class Delta(Funsor):
    """
    Normalized delta distribution binding a single variable.

    :param str name: Name of the bound variable.
    :param Funsor point: Value of the bound variable.
    :param Funsor log_density: Optional log density to be added when evaluating
        at a point. This is needed to make :class:`Delta` closed under
        differentiable substitution.
    """
    def __init__(self, name, point, log_density=0):
        assert isinstance(name, str)
        assert isinstance(point, Funsor)
        assert isinstance(log_density, Funsor)
        assert log_density.output == reals()
        inputs = OrderedDict([(name, point.output)])
        inputs.update(point.inputs)
        inputs.update(log_density.inputs)
        output = reals()
        fresh = frozenset({name})
        bound = frozenset()
        super(Delta, self).__init__(inputs, output, fresh, bound)
        self.name = name
        self.point = point
        self.log_density = log_density

    def eager_subs(self, subs):
        assert len(subs) == 1 and subs[0][0] == self.name
        value = subs[0][1]

        if isinstance(value, (str, Variable)):
            value = to_funsor(value, self.output)
            return Delta(value.name, self.point, self.log_density)

        if not any(d.dtype == 'real' for side in (value, self.point)
                   for d in side.inputs.values()):
            return (value == self.point).all().log() + self.log_density

        # Try to invert the substitution.
        soln = solve(value, self.point)
        if soln is None:
            return None  # lazily substitute
        name, point, log_density = soln
        log_density += self.log_density
        return Delta(name, point, log_density)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if self.name in reduced_vars:
                return to_funsor(ops.log(
                    reduce(ops.mul, [self.inputs[v].dtype for v in reduced_vars - self.fresh], 1)))

        # TODO Implement ops.add to simulate .to_event().

        return None  # defer to default implementation

    def unscaled_sample(self, sampled_vars, sample_inputs):
        return self


@eager.register(Binary, AddOp, Delta, (Funsor, Align))
def eager_add(op, lhs, rhs):
    if lhs.name in rhs.inputs:
        rhs = rhs(**{lhs.name: lhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, SubOp, Delta, (Funsor, Align))
def eager_sub(op, lhs, rhs):
    if lhs.name in rhs.inputs:
        rhs = rhs(**{lhs.name: lhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, AddOp, (Funsor, Align), Delta)
def eager_add(op, lhs, rhs):
    if rhs.name in lhs.inputs:
        lhs = lhs(**{rhs.name: rhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


eager.register(Binary, AddOp, Delta, Reduce)(
    funsor.terms.eager_distribute_other_reduce)
eager.register(Binary, AddOp, Reduce, Delta)(
    funsor.terms.eager_distribute_reduce_other)


@eager.register(Independent, Delta, str, str)
def eager_independent(delta, reals_var, bint_var):
    if delta.name == reals_var or delta.name.startswith(reals_var + "__BOUND"):
        i = Variable(bint_var, delta.inputs[bint_var])
        point = Lambda(i, delta.point)
        if bint_var in delta.log_density.inputs:
            log_density = delta.log_density.reduce(ops.add, bint_var)
        else:
            log_density = delta.log_density * delta.inputs[bint_var].dtype
        return Delta(reals_var, point, log_density)

    return None  # defer to default implementation


@eager.register(Integrate, Delta, Funsor, frozenset)
def eager_integrate(delta, integrand, reduced_vars):
    assert delta.name in reduced_vars
    integrand = Subs(integrand, ((delta.name, delta.point),))
    log_measure = delta.log_density
    reduced_vars -= frozenset([delta.name])
    return Integrate(log_measure, integrand, reduced_vars)


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


@add_metaclass(MultiDeltaMeta)
class MultiDelta(Funsor):
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


@eager.register(Binary, AddOp, Delta, Delta)
def eager_add_delta_delta(op, lhs, rhs):
    if lhs.name == rhs.name:
        raise NotImplementedError
    return MultiDelta(((lhs.name, (lhs.point, lhs.log_density)),)) + \
        MultiDelta(((rhs.name, (rhs.point, rhs.log_density)),))


@eager.register(Binary, AddOp, MultiDelta, Delta)
def eager_add_delta_multidelta(op, lhs, rhs):
    return op(lhs, MultiDelta(((rhs.name, (rhs.point, rhs.log_density)),)))


@eager.register(Binary, AddOp, Delta, MultiDelta)
def eager_add_multidelta_delta(op, lhs, rhs):
    return op(rhs, lhs)


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
        rhs = rhs(**{name: point for name, point in lhs.terms if name in rhs.inputs})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, AddOp, (Funsor, Align), MultiDelta)
def eager_add_funsor_delta(op, lhs, rhs):
    if rhs.fresh.intersection(lhs.inputs):
        lhs = lhs(**{name: point for name, point in rhs.terms if name in lhs.inputs})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Integrate, MultiDelta, Funsor, frozenset)
def eager_integrate(delta, integrand, reduced_vars):
    assert reduced_vars & delta.fresh
    subs = tuple((name, point) for name, (point, log_density) in delta.terms
                 if name in reduced_vars)
    integrand = Subs(integrand, subs)
    log_measure = Subs(delta, subs)
    reduced_vars -= delta.fresh
    return Integrate(log_measure, integrand, reduced_vars)


__all__ = [
    'Delta',
    'MultiDelta',
    'solve',
]
