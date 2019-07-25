import functools

import unification.match
from unification import unify
from unification.variable import isvar

import funsor.ops as ops
from funsor.interpreter import dispatched_interpretation, interpretation
from funsor.terms import Binary, Funsor, Variable, lazy


@dispatched_interpretation
def unify_interpreter(cls, *args):
    result = unify_interpreter.dispatch(cls, *args)
    if result is None:
        result = lazy(cls, *args)
    return result


@unify_interpreter.register(Binary, ops.Op, Funsor, Funsor)
def unify_eq(op, lhs, rhs):
    if op is ops.eq:
        return lhs is rhs  # equality via cons-hashing
    return None


class EqDispatcher(unification.match.Dispatcher):

    resolve = interpretation(unify_interpreter)(unification.match.Dispatcher.resolve)


class EqVarDispatcher(EqDispatcher):

    def __call__(self, *args, **kwargs):
        func, s = self.resolve(args)
        d = dict((k.name if isinstance(k, Variable) else k.token, v) for k, v in s.items())
        return func(**d)


@isvar.register(Variable)
def _isvar_funsor_variable(v):
    return True


@unify.register(Funsor, Funsor, dict)
@interpretation(unify_interpreter)
def unify_funsor(pattern, expr, subs):
    if type(pattern) is not type(expr):
        return False
    return unify(pattern._ast_values, expr._ast_values, subs)


@unify.register(Variable, (Variable, Funsor), dict)
@unify.register((Variable, Funsor), Variable, dict)
def unify_patternvar(pattern, expr, subs):
    subs.update({pattern: expr} if isinstance(pattern, Variable) else {expr: pattern})
    return subs


match_vars = functools.partial(unification.match.match, Dispatcher=EqVarDispatcher)
match = functools.partial(unification.match.match, Dispatcher=EqDispatcher)


__all__ = [
    "match",
    "match_vars",
]
