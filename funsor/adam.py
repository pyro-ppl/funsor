# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor.ops as ops
from funsor.domains import RealsType
from funsor.interpretations import StatefulInterpretation
from funsor.tensor import Tensor
from funsor.terms import Funsor, Reduce
from funsor.util import get_backend


class Adam(StatefulInterpretation):
    """
    Usage::
        with Adam(num_steps=100, lr=0.05)) as optim:
            x = loss.reduce(ops.min)

        print("final loss", x)  # equivalent to loss(**optim.params)
        for name in loss.inputs:
            print(name, optim.params(name))
    """

    def __init__(self, num_steps, **kwargs):
        name = kwargs.pop("name", "adam")
        super().__init__(name)
        self.num_steps = num_steps
        self.log_every = kwargs.pop("log_every", 0)
        self.optim_params = kwargs  # TODO make precise
        self.params = kwargs.pop("params", {})

    def param(self, name, domain=None):
        if name not in self.params:
            if domain is None:
                raise ValueError(f"Unknown param: {name}")
            self.params[name] = self._initialize(name, domain)
        return self.params[name]

    def _initialize(self, name, domain):
        assert isinstance(name, str)
        assert isinstance(domain, RealsType)
        if get_backend() == "torch":
            import torch

            return Tensor(torch.randn(domain.shape, requires_grad=True))
        raise NotImplementedError(f"Unsupported backend {get_backend()}")


@Adam.register(Reduce, ops.MinOp, Funsor, frozenset)
def adam_min(self, op, loss, reduced_vars):
    if get_backend() == "torch":
        import torch

        with torch.enable_grad():
            params = {
                var.name: self.param(var.name, var.output).data
                for var in reduced_vars.intersection(loss.input_vars)
            }
            optimizer = torch.optim.Adam(list(params.values()), **self.optim_params)
            for step in range(self.num_steps):
                optimizer.zero_grad()
                # Note we use v[...] to trigger a shallow copy of the underlying
                # torch.Tensor object so that funsor.Tensor(-) creates a new
                # cons-hashed value and avoids possible downstream memoization
                # (which would be incorrect because underlying data has changed).
                # TODO test that this interacts with cons-hashing correctly
                step_loss = loss(**{k: v[...] for k, v in params.items()}).data
                step_loss.backward()
                optimizer.step()
                if self.log_every and step % self.log_every == 0:
                    print(f"step {step: >6d} loss = {step_loss.data:g}")
    else:
        raise NotImplementedError(f"Unsupported backend {get_backend()}")
    return loss(**params)


@Adam.register(Reduce, ops.MaxOp, Funsor, frozenset)
def adam_max(self, op, loss, reduced_vars):
    return (-loss).reduce(ops.min, reduced_vars)
