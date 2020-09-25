# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import funsor
import funsor.ops as ops
from funsor.domains import RealsType
from funsor.interpreter import StatefulInterpretation


class Adam(StatefulInterpretation):
    """
    Usage::
        with interpretation(AdamInterpreter(lr=0.05)):
            x = loss.reduce(ops.min)
    """
    def __init__(self, num_steps, **kwargs):
        self.num_steps = num_steps
        self.optim_params = kwargs  # TODO make precise

    def __call__(self, cls, *args):
        return self.dispatch(cls, *args)(self, *args)  # may return None

    def initialize(self, name, domain):
        assert isinstance(name, str)
        assert isinstance(domain, RealsType)
        return torch.randn(domain.shape, requires_grad=True)


@Adam.register(funsor.terms.Reduce, funsor.ops.MinOp, funsor.Funsor, frozenset)
@funsor.adjoint.block_adjoint()
def adam_min(self, op, loss, reduced_vars):
    with torch.grad_enabled():  # , funsor.adjoint.adjoint_block():  # TODO define
        params = {var: self.initialize(var, loss.inputs[var])
                  for var in reduced_vars.intersection(loss.inputs)}
        optimizer = torch.optim.Adam(list(params.values()), **self.optim_params)
        for step in range(self.num_steps):
            optimizer.zero_grad()
            # TODO test that this interacts with cons-hashing correctly
            loss(**{k: v[...] for k, v in params.items()}).data.backward()
            optimizer.step()
    return loss(**params)


@Adam.register(funsor.terms.Argreduce, funsor.ops.MinOp, funsor.Funsor, frozenset)
def adam_min(self, op, loss, reduced_vars):
    with torch.grad_enabled():
        params = {var: self.initialize(var, loss.inputs[var])
                  for var in reduced_vars.intersection(loss.inputs)}
        optimizer = torch.optim.Adam(list(params.values()), **self.optim_params)
        for step in range(self.num_steps):
            optimizer.zero_grad()
            # TODO test that this interacts with cons-hashing correctly
            loss(**{k: v[...] for k, v in params.items()}).data.backward()
            optimizer.step()

    # only difference to Reduce is here
    params = {k: ops.detach(v) for k, v in params.items()}
    return funsor.delta.Delta(tuple(params.items())) + loss(**params)
