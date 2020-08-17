# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import io
import pickle

import pytest

from funsor.domains import Bint, Real, Reals, Bint, Reals  # noqa F401


@pytest.mark.parametrize('expr', [
    "Bint[2]",
    "Real",
    "Reals[4]",
    "Reals[3, 2]",
])
def test_pickle(expr):
    x = eval(expr)
    f = io.BytesIO()
    pickle.dump(x, f)
    f.seek(0)
    y = pickle.load(f)
    assert y is x


def test_cache():
    assert Bint[1] is Bint[1]
    assert Real is Reals[()]
    assert Reals[2, 3] is Reals[2, 3]


def test_subclass():
    assert issubclass(Bint, Bint)
    assert issubclass(Bint[1], Bint)
    assert issubclass(Bint[1], Bint[1])
    assert issubclass(Bint[2], Bint)
    assert issubclass(Bint[2], Bint[2])
    assert not issubclass(Bint, Bint[1])
    assert not issubclass(Bint, Bint[2])
    assert not issubclass(Bint[1], Bint[2])
    assert not issubclass(Bint[2], Bint[1])

    assert issubclass(Reals, Reals)
    assert issubclass(Real, Real)
    assert issubclass(Real, Reals)
    assert issubclass(Reals[2], Reals)
    assert issubclass(Reals[2], Reals[2])
    assert not issubclass(Reals, Real)
    assert not issubclass(Reals, Reals[2])
    assert not issubclass(Real, Reals[2])
    assert not issubclass(Reals[2], Real)

    assert not issubclass(Reals, Bint)
    assert not issubclass(Bint, Reals)
    assert not issubclass(Reals[2], Bint[2])
    assert not issubclass(Bint[2], Reals[2])
