# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from multipledispatch import dispatch

from funsor.tensor import tensor_to_funsor
from funsor.terms import to_funsor
from funsor.util import quote

from . import distributions as _
from . import ops as _
from .metadata import MetadataTensor

del _  # flake8


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


__all__ = ["MetadataTensor"]
