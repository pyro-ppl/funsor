from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Domain, reals
from funsor.integrate import Integrate
from funsor.interpreter import debug_logged
from funsor.ops import AddOp, SubOp, TransformOp
from funsor.registry import KeyedRegistry
from funsor.terms import (
    Align,
    Binary,
    Funsor,
    Number,
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


class MultiDelta(Funsor):
    """
    Normalized delta distribution binding multiple variables.
    """
    def __init__(self, terms):
        assert isinstance(terms, tuple) and len(terms) > 0
        inputs = OrderedDict()  # log_density.inputs.copy()
        for name, (point, log_density) in terms:
            assert isinstance(name, str)
            assert isinstance(point, Funsor)
            assert isinstance(log_density, Funsor)
            assert log_density.output == reals()
            assert name not in inputs
            assert name not in point.inputs
            inputs.update({name: point.output})
            inputs.update(point.inputs)

        output = reals()
        fresh = frozenset(name for name, term in terms)
        bound = frozenset()
        super(MultiDelta, self).__init__(inputs, output, fresh, bound)
        self.terms = terms

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.fresh for name in names)
        if not names or names == tuple(n for n, p in self.terms):
            return self

        new_terms = sorted(self.terms, key=lambda t: names.index(t[0]))
        return MultiDelta(new_terms)

    def eager_subs(self, subs):
        terms = OrderedDict(self.terms)
        new_terms = terms.copy()
        log_density = to_funsor(0., self.output)
        for name, value in subs:
            if isinstance(value, Variable):
                new_terms[value.name] = new_terms.pop(name)
                continue

            if not any(d.dtype == 'real' for side in (value, terms[name][0])
                       for d in side.inputs.values()):
                point, point_log_density = new_terms.pop(name)
                log_density += (value == point).all().log() + point_log_density
                continue

            # Try to invert the substitution.
            soln = solve(value, terms[name][0])
            if soln is None:
                return None  # lazily substitute
            new_name, new_point, point_log_density = soln
            old_point, old_point_density = new_terms.pop(name)
            new_terms[new_name] = (new_point, old_point_density + point_log_density)

        return MultiDelta(tuple(new_terms.items())) + log_density if new_terms else log_density

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if reduced_vars - self.fresh and self.fresh - reduced_vars:
                result = self.eager_reduce(op, reduced_vars & self.fresh) if reduced_vars & self.fresh else self
                if result is not self:
                    result = result.eager_reduce(op, reduced_vars - self.fresh) if reduced_vars - self.fresh else self
                    return result if result is not self else None
                return None

            result_terms = [(name, (point, log_density)) for name, (point, log_density) in self.terms
                            if name not in reduced_vars]

            result_terms, scale = [], Number(0)
            for name, (point, log_density) in self.terms:
                if name in reduced_vars:
                    # XXX obscenely wasteful - need a lazy Zero term
                    if point.inputs:
                        scale += (point == point).all().log()
                    if log_density.inputs:
                        scale += log_density * 0.
                else:
                    result_terms.append((name, (point, log_density)))

            result = MultiDelta(tuple(result_terms)) + scale if result_terms else scale
            return result.reduce(op, reduced_vars - self.fresh)

        if op is ops.add:
            raise NotImplementedError("TODO Implement ops.add to simulate .to_event().")

        return None  # defer to default implementation

    def unscaled_sample(self, sampled_vars, sample_inputs):
        if sampled_vars <= self.fresh:
            return self
        raise NotImplementedError("TODO implement sample for particle indices")


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
        rhs = rhs(**{name: point for name, (point, log_density) in lhs.terms if name in rhs.inputs})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, AddOp, (Funsor, Align), MultiDelta)
def eager_add_funsor_delta(op, lhs, rhs):
    if rhs.fresh.intersection(lhs.inputs):
        lhs = lhs(**{name: point for name, (point, log_density) in rhs.terms if name in lhs.inputs})
        return op(lhs, rhs)

    return None


@eager.register(Integrate, MultiDelta, Funsor, frozenset)
def eager_integrate(delta, integrand, reduced_vars):
    if not reduced_vars & delta.fresh:
        return None
    subs = tuple((name, point) for name, (point, log_density) in delta.terms
                 if name in reduced_vars)
    new_integrand = Subs(integrand, subs)
    new_log_measure = Subs(delta, subs)
    result = Integrate(new_log_measure, new_integrand, reduced_vars - delta.fresh)
    return result


__all__ = [
    'Delta',
    'MultiDelta',
    'solve',
]
