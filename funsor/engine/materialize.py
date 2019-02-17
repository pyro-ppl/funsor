from __future__ import absolute_import, division, print_function

from funsor.engine.interpreter import eval
from funsor.handlers import OpRegistry
from funsor.interpretations import Eager
from funsor.terms import Variable
from funsor.torch import Arange


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
    x = Eager(eval)(x)
    return x


__all__ = [
    'materialize',
]
