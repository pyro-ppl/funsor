# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import torch
from multipledispatch import dispatch

from funsor.constant import Constant
from funsor.tensor import tensor_to_funsor
from funsor.terms import to_funsor
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
    out.append((indent, "torch.tensor({}, dtype={})".format(repr(x.tolist()), x.dtype)))


@to_funsor.register(ProvenanceTensor)
def provenance_to_funsor(x, output=None, dim_to_name=None):
    if isinstance(x, ProvenanceTensor):
        ret = to_funsor(x._t, output=output, dim_to_name=dim_to_name)
        return Constant(OrderedDict(x._provenance), ret)


to_funsor.register(torch.Tensor)(tensor_to_funsor)


@dispatch(torch.Tensor, torch.Tensor, [float])
def allclose(a, b, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, b, rtol=rtol, atol=atol)
