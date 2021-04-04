# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from math import pi

from funsor.factory import Bound, Fresh, Has, make_funsor
from funsor.interpretations import StatefulInterpretation
from funsor.tensor import Tensor
from funsor.terms import Funsor, Variable
from funsor.util import get_backend

from . import ops


@make_funsor
def InverseLaplace(
    F: Has[{"s"}], t: Funsor, s: Bound  # noqa: F821
) -> Fresh[lambda F: F]:
    """
    Inverse Laplace transform of function F.

    :param F: function of s.
    :param t: times at which to evaluate the inverse Laplace transformation of F.
    :param s: s Variable.
    """
    return None


class Talbot(StatefulInterpretation):
    """
    Talbot's method for numerical inversion of the Laplace transform.

    Reference
    Abate, Joseph, and Ward Whitt. "A Unified Framework for Numerically
    Inverting Laplace Transforms." INFORMS Journal of Computing, vol. 18.4
    (2006): 408-421. Print. (http://www.columbia.edu/~ww2040/allpapers.html)

    Implementation here is adapted from the MATLAB implementation of the algorithm by
    Tucker McClure (2021). Numerical Inverse Laplace Transform
    (https://www.mathworks.com/matlabcentral/fileexchange/39035-numerical-inverse-laplace-transform),
    MATLAB Central File Exchange. Retrieved April 4, 2021.

    :param num_steps: number of terms to sum for each t.
    """

    def __init__(self, num_steps):
        super().__init__("talbot")
        self.N = num_steps


@Talbot.register(InverseLaplace, Funsor, Funsor, Variable)
def talbot(self, F, t, s):
    if get_backend() == "torch":
        import torch

        k = torch.arange(1, self.N)
        delta = torch.zeros(self.N, dtype=torch.complex64)
        delta[0] = 2 * self.N / 5
        delta[1:] = 2 * pi / 5 * k * (1 / (pi / self.N * k).tan() + 1j)

        gamma = torch.zeros(self.N, dtype=torch.complex64)
        gamma[0] = 0.5 * delta[0].exp()
        gamma[1:] = (
            1
            + 1j * pi / self.N * k * (1 + 1 / (pi / self.N * k).tan() ** 2)
            - 1j / (pi / self.N * k).tan()
        ) * delta[1:].exp()

        delta = Tensor(delta)["num_steps"]
        gamma = Tensor(gamma)["num_steps"]
        ilt = 0.4 / t * (gamma * F(**{s.name: delta / t})).reduce(ops.add, "num_steps")

        return Tensor(ilt.data.real, ilt.inputs)
    else:
        raise NotImplementedError(f"Unsupported backend {get_backend()}")


__all__ = [
    "Talbot",
]
