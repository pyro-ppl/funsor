# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Talbot's method for numerical inversion of the Laplace transform
=========================================================================

"""

import argparse

import torch

import funsor
import funsor.ops as ops
from funsor.adam import Adam
from funsor.domains import Real
from funsor.inverse_laplace import InverseLaplace, Talbot
from funsor.tensor import Tensor
from funsor.terms import Variable


def main(args):
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

    with Adam(
        args.num_steps,
        lr=args.learning_rate,
        log_every=args.log_every,
        params=init_params,
    ) as optim:
        with Talbot(num_steps=args.talbot_num_steps):
            loss.reduce(ops.min, {"rate", "nsteps"})

    # Fit curve.
    with Talbot(num_steps=args.talbot_num_steps):
        fit = pred(rate=optim.param("rate"), nsteps=optim.param("nsteps"))

    print(f"Data\n{data}")
    print(f"Fit curve\n{fit}")
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
