from __future__ import absolute_import, division, print_function

from unification import unify
from unification.variable import isvar

import funsor.ops as ops
from funsor.interpreter import interpretation
from funsor.terms import Binary, Funsor, Variable, lazy


@lazy.register(Binary, ops.Op, Funsor, Funsor)
def lazy_eq(op, lhs, rhs):
    if op is ops.eq:
        return lhs is rhs  # equality via cons-hashing
    if op is ops.ne:
        return lhs is not rhs
    return None


@isvar.register(Variable)
def _isvar_funsor_variable(v):
    return True


@unify.register(Funsor, Funsor, dict)
@interpretation(lazy)
def unify_funsor(pattern, expr, subs):
    if type(pattern) is not type(expr):
        return False
    return unify(pattern._ast_values, expr._ast_values, subs)


@unify.register(Variable, Funsor, dict)
def unify_patternvar(pattern, expr, subs):
    subs.update({pattern: expr})
    return subs
