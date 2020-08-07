# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from multipledispatch import dispatch

import funsor.torch.distributions  # noqa: F401
import funsor.torch.ops  # noqa: F401
import funsor.ops as ops
from funsor.adjoint import adjoint_ops
from funsor.interpreter import children, recursion_reinterpret
from funsor.terms import Funsor, to_funsor
from funsor.tensor import Tensor, tensor_to_funsor
from funsor.util import quote


@adjoint_ops.register(Tensor, ops.AssociativeOp, ops.AssociativeOp, Funsor, torch.Tensor, tuple, object)
def adjoint_tensor(adj_redop, adj_binop, out_adj, data, inputs, dtype):
    return {}


@recursion_reinterpret.register(torch.Tensor)
@recursion_reinterpret.register(torch.nn.Module)
def recursion_reinterpret_ground(x):
    return x


@children.register(torch.Tensor)
@children.register(torch.nn.Module)
def _children_ground(x):
    return ()


@quote.register(torch.Tensor)
def _quote(x, indent, out):
    """
    Work around PyTorch not supporting reproducible repr.
    """
    out.append((indent, "torch.tensor({}, dtype={})".format(repr(x.tolist()), x.dtype)))


to_funsor.register(torch.Tensor)(tensor_to_funsor)


@dispatch(torch.Tensor, torch.Tensor, [float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, b, rtol=rtol, atol=atol)
