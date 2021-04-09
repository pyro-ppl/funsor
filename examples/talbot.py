# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Talbot's method for numerical inversion of the Laplace transform
=========================================================================

"""

import argparse
import math

import funsor
import funsor.ops as ops
from funsor.adam import Adam
from funsor.domains import Real
from funsor.factory import Bound, Fresh, Has, make_funsor
from funsor.interpretations import StatefulInterpretation
from funsor.tensor import Tensor
from funsor.terms import Funsor, Variable
from funsor.util import backends_supported


@make_funsor
def InverseLaplace(
    F: Has[{"s"}], t: Funsor, s: Bound  # noqa: F821
) -> Fresh[lambda F: F]:
    """
    Inverse Laplace transform of function F(s).

    There is no closed-form solution for arbitrary F(s). However, we can
    resort to numerical approximations which we store in new interpretations.
    For example, see Talbot's method below.

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
        self.num_steps = num_steps


@Talbot.register(InverseLaplace, Funsor, Funsor, Variable)
@backends_supported("torch")
def talbot(self, F, t, s):
    import torch

    k = torch.arange(1, self.num_steps)
    delta = torch.zeros(self.num_steps, dtype=torch.complex64)
    delta[0] = 2 * self.num_steps / 5
    delta[1:] = 2 * math.pi / 5 * k * (1 / (math.pi / self.num_steps * k).tan() + 1j)

    gamma = torch.zeros(self.num_steps, dtype=torch.complex64)
    gamma[0] = 0.5 * delta[0].exp()
    gamma[1:] = (
        1
        + 1j
        * math.pi
        / self.num_steps
        * k
        * (1 + 1 / (math.pi / self.num_steps * k).tan() ** 2)
        - 1j / (math.pi / self.num_steps * k).tan()
    ) * delta[1:].exp()

    delta = Tensor(delta)["num_steps"]
    gamma = Tensor(gamma)["num_steps"]
    ilt = 0.4 / t * (gamma * F(**{s.name: delta / t})).reduce(ops.add, "num_steps")

    return Tensor(ilt.data.real, ilt.inputs)


def main(args):
    """
    Reference for the n-step sequential model used here:

    Aaron L. Lucius et al (2003).
    "General Methods for Analysis of Sequential ‘‘n-step’’ Kinetic Mechanisms:
    Application to Single Turnover Kinetics of Helicase-Catalyzed DNA Unwinding"
    https://www.sciencedirect.com/science/article/pii/S0006349503746487
    """
    import torch

    funsor.set_backend("torch")

    # Problem definition.
    true_rate = 20
    true_nsteps = 4
    rate = Variable("rate", Real)
    nsteps = Variable("nsteps", Real)
    s = Variable("s", Real)
    time = Tensor(torch.arange(0.04, 1.04, 0.04))["timepoint"]
    noise = Tensor(torch.randn(time.inputs["timepoint"].size))["timepoint"] / 500
    data = (
        Tensor(1 - torch.igammac(torch.tensor(true_nsteps), true_rate * time.data))[
            "timepoint"
        ]
        + noise
    )
    F = rate ** nsteps / (s * (rate + s) ** nsteps)
    # Inverse Laplace.
    pred = InverseLaplace(F, time, "s")

    # Loss function.
    loss = (pred - data).abs().reduce(ops.add, "timepoint")
    init_params = {
        "rate": Tensor(torch.tensor(5.0, requires_grad=True)),
        "nsteps": Tensor(torch.tensor(2.0, requires_grad=True)),
    }

    with Talbot(num_steps=args.talbot_num_steps):
        # Fit the data
        with Adam(
            args.num_steps,
            lr=args.learning_rate,
            log_every=args.log_every,
            params=init_params,
        ) as optim:
            loss.reduce(ops.min, {"rate", "nsteps"})
        # Fitted curve.
        fitted_curve = pred(rate=optim.param("rate"), nsteps=optim.param("nsteps"))

    print(f"Data\n{data}")
    print(f"Fit curve\n{fitted_curve}")
    print(f"True rate\n{true_rate}")
    print("Learned rate\n{}".format(optim.param("rate").item()))
    print(f"True number of steps\n{true_nsteps}")
    print("Learned number of steps\n{}".format(optim.param("nsteps").item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Numerical inversion of the Laplace transform using Talbot's method"
    )
    parser.add_argument("-N", "--talbot-num-steps", type=int, default=32)
    parser.add_argument("-n", "--num-steps", type=int, default=501)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    main(args)
