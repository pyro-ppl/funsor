# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import io
import pickle

import pytest

from funsor.domains import bint, reals  # noqa F401


@pytest.mark.parametrize('expr', [
    "bint(2)",
    "reals()",
    "reals(4)",
    "reals(3, 2)",
])
def test_pickle(expr):
    x = eval(expr)
    f = io.BytesIO()
    pickle.dump(x, f)
    f.seek(0)
    y = pickle.load(f)
    y is x
