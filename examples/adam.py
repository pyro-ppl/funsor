# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Adam optimizer
=======================

"""

import argparse

import torch

import funsor
import funsor.ops as ops
from funsor.adam import Adam
from funsor.domains import Real, Reals
from funsor.tensor import Tensor
from funsor.terms import Variable


def main(args):
    funsor.set_backend("torch")

    # Problem definition.
    N = 100
    P = 10
    data = Tensor(torch.randn(N, P))["n"]
    true_weight = Tensor(torch.randn(P))
    true_bias = Tensor(torch.randn(()))
    truth = true_bias + true_weight @ data

    # Model.
    weight = Variable("weight", Reals[P])
    bias = Variable("bias", Real)
    pred = bias + weight @ data
    loss = (pred - truth).abs().reduce(ops.add, "n")

    # Inference.
    with Adam(args.num_steps, lr=args.learning_rate, log_every=args.log_every) as optim:
        loss.reduce(ops.min, {"weight", "bias"})

    print(f"True bias\n{true_bias}")
    print("Learned bias\n{}".format(optim.param("bias")))
    print(f"True weight\n{true_weight}")
    print("Learned weight\n{}".format(optim.param("weight")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear regression example using Adam interpretation"
    )
    parser.add_argument("-P", "--num-features", type=int, default=10)
    parser.add_argument("-N", "--num-data", type=int, default=100)
    parser.add_argument("-n", "--num-steps", type=int, default=201)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    main(args)
