# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import re

from jax.core import Tracer
from jax.numpy import ndarray

from funsor.tensor import tensor_to_funsor
from funsor.terms import to_funsor
from funsor.util import quote

from . import distributions as _
from . import ops as _

del _  # flake8


to_funsor.register(ndarray)(tensor_to_funsor)
to_funsor.register(Tracer)(tensor_to_funsor)


@quote.register(ndarray)
def _quote(x, indent, out):
    """
    Work around JAX's ndarray not supporting reproducible repr.
    """
    # After JAX 0.4, jnp.ones(3) is no longer a DeviceArray, but an ndarray.
    # In addition, a tracer is also an ndarray - so we need to handler it
    # separately here.
    if isinstance(x, Tracer):
        # Default implementation.
        line = re.sub("\n\\s*", " ", repr(x))
        out.append((indent, line))
        return
    if x.size >= quote.printoptions["threshold"]:
        data = "..." + " x ".join(str(d) for d in x.shape) + "..."
    else:
        data = repr(x.copy().tolist())
    out.append((indent, f"np.array({data}, dtype=np.{x.dtype})"))
