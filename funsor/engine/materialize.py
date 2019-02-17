from __future__ import absolute_import, division, print_function

from funsor.engine.opteinsum_engine import EagerEval
from funsor.handlers import OpRegistry
from funsor.terms import Variable
from funsor.torch import Arange
from funsor.engine.interpreter import eval


class Materialize(OpRegistry):
    pass


@Materialize.register(Variable)
def _materialize_variable(name, size):
    if isinstance(size, int):
        return Arange(name, size)
    else:
        return Variable(name, size)


def materialize(x):
    x = Materialize(eval)(x)
    x = EagerEval(eval)(x)
    return x


__all__ = [
    'materialize',
]
