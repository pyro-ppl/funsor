# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from collections import OrderedDict

import torch

import funsor
import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import Bint
from funsor.tensor import Tensor
from funsor.terms import Number, Stack, Variable


def Uniform(components):
    components = tuple(components)
    size = len(components)
    if size == 1:
        return components[0]
    var = Variable("v", Bint[size])
    return Stack(var.name, components).reduce(ops.logaddexp, var.name) - math.log(size)


# @of_shape(*([Bint[2]] * size))
def model(size, position=0):
    if size == 1:
        name = str(position)
        return Uniform((Delta(name, Number(0, 2)), Delta(name, Number(1, 2))))
    return Uniform(
        model(t, position) + model(size - t, t + position) for t in range(1, size)
    )


def main(args):
    funsor.set_backend("torch")
    torch.manual_seed(args.seed)

    print_ = print if args.verbose else lambda msg: None
    print_("Data:")
    data = torch.distributions.Categorical(torch.ones(2)).sample((args.size,))
    assert data.shape == (args.size,)
    data = Tensor(data, OrderedDict(i=Bint[args.size]), dtype=2)
    print_(data)

    print_("Model:")
    m = model(args.size)
    print_(m.pretty())

    print_("Eager log_prob:")
    obs = {str(i): data(i) for i in range(args.size)}
    log_prob = m(**obs)
    print_(log_prob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCFG example")
    parser.add_argument("-s", "--size", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
