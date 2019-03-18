from __future__ import absolute_import, division, print_function

from funsor.domains import reals
from funsor.interpreter import interpretation
from funsor.terms import Number, Variable, reflect

from unification import unify


def test_matcher_binary():
    with interpretation(reflect):
        pattern = Variable('a', reals()) + Number(2.) * Variable('b', reals())
        expr = Number(1.) + Number(2.) * (Number(3.) - Number(4.))

    subs = unify(pattern, expr)
    print(subs, pattern(**{k.name: v for k, v in subs.items()}))
    assert subs is not False
