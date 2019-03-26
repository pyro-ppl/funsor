from __future__ import absolute_import, division, print_function

import argparse
import math
from collections import OrderedDict

import torch

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import bint
from funsor.terms import Number, Stack, Variable
from funsor.torch import Tensor


def Uniform(components):
    components = tuple(components)
    size = len(components)
    if size == 1:
        return components[0]
    var = Variable('v', bint(size))
    return (Stack(components, var.name).reduce(ops.logaddexp, var.name)
            - math.log(size))


# @of_shape(*([bint(2)] * size))
def model(size, position=0):
    if size == 1:
        name = str(position)
        return Uniform((Delta(name, Number(0, 2)),
                        Delta(name, Number(1, 2))))
    return Uniform(model(t, position) +
                   model(size - t, t + position)
                   for t in range(1, size))


def main(args):
    torch.manual_seed(args.seed)

    print_ = print if args.verbose else lambda msg: None
    print_('Data:')
    data = torch.distributions.Categorical(torch.ones(2)).sample((args.size,))
    assert data.shape == (args.size,)
    data = Tensor(data, OrderedDict(i=bint(args.size)), dtype=2)
    print_(data)

    print_('Model:')
    m = model(args.size)
    print_(m.pretty())

    print_('Eager log_prob:')
    obs = {str(i): data(i) for i in range(args.size)}
    log_prob = m(**obs)
    print_(log_prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCFG example")
    parser.add_argument("-s", "--size", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()
    main(args)
