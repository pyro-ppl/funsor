# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import torch
from multipledispatch import dispatch

from funsor.constant import Constant
from funsor.tensor import tensor_to_funsor
from funsor.terms import to_data, to_funsor
from funsor.torch.provenance import ProvenanceTensor
from funsor.util import quote

from . import distributions as _
from . import ops as _

del _  # flake8


@quote.register(torch.Tensor)
def _quote(x, indent, out):
    """
    Work around PyTorch not supporting reproducible repr.
    """
    if x.numel() >= quote.printoptions["threshold"]:
        data = "..." + " x ".join(str(int(d)) for d in x.shape) + "..."
    else:
        data = repr(x.tolist())
    out.append((indent, f"torch.tensor({data}, dtype={x.dtype})"))


@to_funsor.register(ProvenanceTensor)
def provenance_to_funsor(x, output=None, dim_to_name=None):
    ret = to_funsor(x._t, output=output, dim_to_name=dim_to_name)
    return Constant(OrderedDict(x._provenance), ret)


@to_data.register(Constant)
def constant_to_data(x, name_to_dim=None):
    data = to_data(x.arg, name_to_dim=name_to_dim)
    return ProvenanceTensor(data, provenance=frozenset(x.const_inputs.items()))


to_funsor.register(torch.Tensor)(tensor_to_funsor)


@dispatch(torch.Tensor, torch.Tensor, [float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, b, rtol=rtol, atol=atol)
