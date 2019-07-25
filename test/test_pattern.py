from unification import unify

from funsor.domains import reals
from funsor.interpreter import interpretation, reinterpret
from funsor.pattern import match, match_vars, unify_interpreter
from funsor.terms import Number, Variable, lazy


def test_unify_binary():
    with interpretation(lazy):
        pattern = Variable('a', reals()) + Number(2.) * Variable('b', reals())
        expr = Number(1.) + Number(2.) * (Number(3.) - Number(4.))

    subs = unify(pattern, expr)
    print(subs, pattern(**{k.name: v for k, v in subs.items()}))
    assert subs is not False

    with interpretation(unify_interpreter):
        assert unify((pattern,), (expr,)) is not False


def test_match_binary():
    with interpretation(lazy):
        pattern = Variable('a', reals()) + Number(2.) * Variable('b', reals())
        expr = Number(1.) + Number(2.) * (Number(3.) - Number(4.))

    @match_vars(pattern)
    def expand_2_vars(a, b):
        return a + b + b

    @match(pattern)
    def expand_2_walk(x):
        return x.lhs + x.rhs.rhs + x.rhs.rhs

    eager_val = reinterpret(expr)
    lazy_val = expand_2_vars(expr)
    assert eager_val == reinterpret(lazy_val)

    lazy_val_2 = expand_2_walk(expr)
    assert eager_val == reinterpret(lazy_val_2)
