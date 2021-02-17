# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpy as np
import pytest

from funsor import ops
from funsor.cnf import Contraction
from funsor.domains import Bint, Real
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.interpretations import eager, lazy, moment_matching, normalize, reflect
from funsor.interpreter import reinterpret
from funsor.montecarlo import MonteCarlo
from funsor.tensor import Tensor
from funsor.terms import Unary, Variable
from funsor.testing import assert_close, random_tensor


@pytest.mark.parametrize(
    "interp",
    [
        reflect,
        lazy,
        normalize,
        eager,
        moment_matching,
        MonteCarlo(rng_key=np.array([0, 0], dtype=np.uint32)),
    ],
)
def test_integrate(interp):
    log_measure = random_tensor(OrderedDict([("i", Bint[2]), ("j", Bint[3])]))
    integrand = random_tensor(OrderedDict([("j", Bint[3]), ("k", Bint[4])]))
    with interp:
        Integrate(log_measure, integrand, {"i", "j", "k"})


def test_syntactic_sugar():
    i = Variable("i", Bint[3])
    log_measure = random_tensor(OrderedDict(i=Bint[3]))
    integrand = random_tensor(OrderedDict(i=Bint[3]))
    expected = (log_measure.exp() * integrand).reduce(ops.add, "i")
    assert_close(Integrate(log_measure, integrand, "i"), expected)
    assert_close(Integrate(log_measure, integrand, {"i"}), expected)
    assert_close(Integrate(log_measure, integrand, frozenset(["i"])), expected)
    assert_close(Integrate(log_measure, integrand, i), expected)
    assert_close(Integrate(log_measure, integrand, {i}), expected)
    assert_close(Integrate(log_measure, integrand, frozenset([i])), expected)


def test_gaussian_integrate_pattern():
    torch = pytest.importorskip("torch")
    with reflect:
        x = Contraction(
            ops.add,
            ops.mul,
            frozenset({Variable("x__BOUND_11", Real)}),
            (
                Unary(
                    ops.exp,
                    Gaussian(
                        torch.tensor([0.0], dtype=torch.float32),
                        torch.tensor([[1.0]], dtype=torch.float32),
                        (
                            (
                                "x__BOUND_11",
                                Real,
                            ),
                        ),
                    ),
                ),
                Contraction(
                    ops.nullop,
                    ops.add,
                    frozenset(),
                    (
                        Tensor(
                            torch.tensor(
                                [
                                    0.9189385175704956,
                                    0.9189385175704956,
                                    0.9189385175704956,
                                ],
                                dtype=torch.float32,
                            ),  # noqa
                            (
                                (
                                    "plate_outer__BOUND_12",
                                    Bint[
                                        3,
                                    ],
                                ),
                            ),
                            "real",
                        ),
                        Gaussian(
                            torch.tensor([-0.0], dtype=torch.float32),
                            torch.tensor([[-1.0]], dtype=torch.float32),
                            (
                                (
                                    "x__BOUND_11",
                                    Real,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    y = reinterpret(x)
    assert isinstance(y, Tensor)
