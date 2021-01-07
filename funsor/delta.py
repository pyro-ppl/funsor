# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Domain, Real
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
    Unary,
    Variable,
    eager,
    to_funsor
)


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


class DeltaMeta(FunsorMeta):
    """
    Makes Delta less of a pain to use by supporting Delta(name, point, log_density)
    """
    def __call__(cls, *args):
        if len(args) > 1:
            assert len(args) == 2 or len(args) == 3
            assert isinstance(args[0], str) and isinstance(args[1], Funsor)
            args = args + (Number(0.),) if len(args) == 2 else args
            args = (((args[0], (to_funsor(args[1]), to_funsor(args[2]))),),)
        assert isinstance(args[0], tuple)
        return super().__call__(args[0])


class Delta(Funsor, metaclass=DeltaMeta):
    """
    Normalized delta distribution binding multiple variables.
    """
    def __init__(self, terms):
        assert isinstance(terms, tuple) and len(terms) > 0
        inputs = OrderedDict()
        for name, (point, log_density) in terms:
            assert isinstance(name, str)
            assert isinstance(point, Funsor)
            assert isinstance(log_density, Funsor)
            assert log_density.output == Real
            assert name not in inputs
            assert name not in point.inputs
            inputs.update({name: point.output})
            inputs.update(point.inputs)

        output = Real
        fresh = frozenset(name for name, term in terms)
        bound = {}
        super(Delta, self).__init__(inputs, output, fresh, bound)
        self.terms = terms

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.fresh for name in names)
        if not names or names == tuple(n for n, p in self.terms):
            return self

        new_terms = sorted(self.terms, key=lambda t: names.index(t[0]))
        return Delta(new_terms)

    def eager_subs(self, subs):
        terms = OrderedDict(self.terms)
        new_terms = terms.copy()
        log_density = Number(0)
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

        return Delta(tuple(new_terms.items())) + log_density if new_terms else log_density

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if op is ops.logaddexp:
            if reduced_vars - self.fresh and self.fresh - reduced_vars:
                result = self
                if not reduced_vars.isdisjoint(self.fresh):
                    result = result.eager_reduce(op, reduced_vars & self.fresh)
                    if result is not self:
                        if not reduced_vars.issubset(self.fresh):
                            result = result.eager_reduce(op, reduced_vars - self.fresh)
                            if result is not self:
                                return result
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

            result = Delta(tuple(result_terms)) + scale if result_terms else scale
            return result.reduce(op, reduced_vars - self.fresh)

        if op is ops.add:
            raise NotImplementedError("TODO Implement ops.add to simulate .to_event().")

        return None  # defer to default implementation

    def unscaled_sample(self, sampled_vars, sample_inputs, rng_key=None):
        return self


@eager.register(Binary, AddOp, Delta, Delta)
def eager_add_multidelta(op, lhs, rhs):
    if lhs.fresh.intersection(rhs.inputs):
        return eager_add_delta_funsor(op, lhs, rhs)

    if rhs.fresh.intersection(lhs.inputs):
        return eager_add_funsor_delta(op, lhs, rhs)

    return Delta(lhs.terms + rhs.terms)


@eager.register(Binary, (AddOp, SubOp), Delta, (Funsor, Align))
def eager_add_delta_funsor(op, lhs, rhs):
    if lhs.fresh.intersection(rhs.inputs):
        rhs = rhs(**{name: point for name, (point, log_density) in lhs.terms if name in rhs.inputs})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, AddOp, (Funsor, Align), Delta)
def eager_add_funsor_delta(op, lhs, rhs):
    if rhs.fresh.intersection(lhs.inputs):
        lhs = lhs(**{name: point for name, (point, log_density) in rhs.terms if name in lhs.inputs})
        return op(lhs, rhs)

    return None


@eager.register(Independent, Delta, str, str, str)
def eager_independent_delta(delta, reals_var, bint_var, diag_var):
    for i, (name, (point, log_density)) in enumerate(delta.terms):
        if name == diag_var:
            bv = Variable(bint_var, delta.inputs[bint_var])
            point = Lambda(bv, point)
            if bint_var in log_density.inputs:
                log_density = log_density.reduce(ops.add, bint_var)
            else:
                log_density = log_density * delta.inputs[bint_var].dtype
            new_terms = delta.terms[:i] + ((reals_var, (point, log_density)),) + delta.terms[i+1:]
            return Delta(new_terms)

    return None


__all__ = [
    'Delta',
    'solve',
]
