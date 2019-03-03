from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.optim import Adam


class Y(nn.Module):
    def __init__(self, f, approx_fix_f):
        assert callable(f)
        assert isinstance(approx_fix_f, nn.Module)
        self.f = f
        self.approx_fix_f = approx_fix_f
        self.buffers.loss = 0  # FIXME

    def approx_f(self, fn, *args):
        return self.approx_fix_f(*args)

    def forward(self, *args):
        # TODO Unroll to multiple depths, gathering loss at each depth.
        value0 = self.approx_fix_f(None, *args)
        value1 = self.f(self.approx_f, *args)
        self.buffers.loss = (value0 - value1).abs().sum()
        return value1


def main(args):

    def f(f, x):
        return 0.5 * x

    def loss_fn(data, value):
        return (data - value).abs().sum()

    data = torch.randn(100)

    Yf = Y(f)

    optim = Adam(lr=args.learning_rate)
    for step in range(args.num_steps):
        value = Yf(data)
        main_loss = loss_fn(data, value)
        loss = main_loss + Yf.buffers.loss  # FIXME
        loss.backward()
        optim.step(Yf.params())
