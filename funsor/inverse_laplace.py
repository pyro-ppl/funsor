# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from functools import reduce

from multipledispatch.variadic import Variadic

from funsor.cnf import Contraction
from funsor.factory import Bound, Fresh, Has, make_funsor
from funsor.integrate import Integrate
from funsor.interpretations import StatefulInterpretation
from funsor.ops import AddOp, LogaddexpOp, NullOp
from funsor.tensor import Tensor
from funsor.terms import Funsor, Number, Reduce, Variable
from funsor.util import get_backend

from . import ops


@make_funsor
def InverseLaplace(
    F: Has[{"s"}], t: Funsor, s: Bound  # noqa: F821
) -> Fresh[lambda F: F]:
    return None


class Talbot(StatefulInterpretation):
    """
    Talbot suggested that the Bromwich line be deformed into a contour that begins
    and ends in the left half plane, i.e., z \to \infty at both ends.
    Due to the exponential factor the integrand decays rapidly
    on such a contour. In such situations the trapezoidal rule converge
    extraordinarily rapidly.

    Reference
    L.N.Trefethen, J.A.C.Weideman, and T.Schmelzer. Talbot quadratures
    and rational approximations. BIT. Numerical Mathematics,
    46(3):653 670, 2006.

    :param Funsor guide: A guide or proposal funsor.
    :param frozenset approx_vars: The variables being integrated.
    """

    def __init__(self, num_steps):
        super().__init__("talbot")
        self.num_steps = num_steps


@Talbot.register(InverseLaplace, Funsor, Funsor, Variable)
def talbot(self, F, t, s):
    if get_backend() == "torch":
        import torch

        #  % Calculate gamma for every index.
        #  gamma = zeros(1, M);
        #  gamma(1) = 0.5*exp(delta(1));
        #  gamma(2:end) =    (1 + 1i*pi/M*k.*(1+cot(pi/M*k).^2)-1i*cot(pi/M*k))...
        #                 .* exp(delta(2:end));
        #
        #  % Make a mesh so we can do this entire calculation across all k for all
        #  % given times without a single loop (it's faster this way).
        #  [delta_mesh, t_mesh] = meshgrid(delta, t);
        #  gamma_mesh = meshgrid(gamma, t);
        #
        #  % Finally, calculate the inverse Laplace transform for each given time.
        #  ilt = 0.4./t .* sum(real(   gamma_mesh ...
        #                           .* arrayfun(f_s, delta_mesh./t_mesh)), 2);

        k = torch.arange(1, self.num_steps)
        delta = torch.zeros(self.num_steps, dtype=torch.complex64)
        delta[0] = 2 * self.num_steps / 5
        delta[1:] = (
            2
            * math.pi
            / 5
            * k
            * (1 / torch.tan(math.pi / self.num_steps * k) + torch.tensor(1j))
        )

        gamma = torch.zeros(self.num_steps, dtype=torch.complex64)
        gamma[0] = 0.5 * delta[1].exp()
        gamma[1:] = (1 + torch.tensor(1j) * math.pi / self.num_steps * k * (
            1 + 1 / torch.tan(math.pi / self.num_steps * k)**2)
            - torch.tensor(1j) / torch.tan(math.pi / self.num_steps * k)) * delta[1:].exp()
        delta_mesh, t_mesh = torch.meshgrid(delta, t.data)
        gamma_mesh = torch.meshgrid(gamma, t.data)[0]
        delta_mesh = Tensor(delta_mesh)[("num_steps",) + tuple(t.inputs)]
        gamma_mesh = Tensor(gamma_mesh)[("num_steps",) + tuple(t.inputs)]
        t_mesh = Tensor(t_mesh)[("num_steps",) + tuple(t.inputs)]
        ilt = 0.4 / t * (gamma_mesh * F(**{s.name: delta_mesh/t_mesh})).reduce(ops.add, "num_steps")
        

        #   Initiate the stepsize
        h = 2 * math.pi / self.num_steps

        #   Shift contour to the right in case there is a pole on the positive real axis : Note the contour will
        #   not be optimal since it was originally devoloped for function with
        #   singularities on the negative real axis
        #   For example take F(s) = 1/(s-1), it has a pole at s = 1, the contour needs to be shifted with one
        #   unit, i.e shift  = 1. But in the test example no shifting is necessary

        shift = 0.0
        ans = 0.0

        # The for loop is evaluating the Laplace inversion at each point theta
        # which is based on the trapezoidal rule
        breakpoint()
        for k in range(0, self.num_steps):
            theta = -math.pi + (k + 0.5) * h
            z = shift + self.num_steps / t * (
                0.5017 * theta / math.tan(0.6407 * theta)
                - 0.6122
                + Tensor(torch.tensor(0.2645j)) * theta
            )
            dz = (
                self.num_steps
                / t
                * (
                    -0.5017 * 0.6407 * theta / math.sin(0.6407 * theta) ** 2
                    + 0.5017 / math.tan(0.6407 * theta)
                    + Tensor(torch.tensor(0.2645j))
                )
            )
            ans = ans + (z * t).exp() * F(**{s.name: z}) * dz

        result = (h / (Tensor(torch.tensor(2j)) * math.pi)) * ans
        return Tensor(result.data.real, result.inputs)
    else:
        raise NotImplementedError(f"Unsupported backend {get_backend()}")


__all__ = [
    "Talbot",
]
