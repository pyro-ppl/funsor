# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax.core import Tracer
from jax.interpreters.xla import DeviceArray

import funsor.jax.distributions  # noqa: F401
import funsor.jax.ops  # noqa: F401
from funsor.interpreter import children, recursion_reinterpret
from funsor.tensor import tensor_to_funsor
from funsor.terms import to_funsor
from funsor.util import quote


@recursion_reinterpret.register(DeviceArray)
@recursion_reinterpret.register(Tracer)
def _recursion_reinterpret_ground(x):
    return x


@children.register(DeviceArray)
@children.register(Tracer)
def _children_ground(x):
    return ()


to_funsor.register(DeviceArray)(tensor_to_funsor)
to_funsor.register(Tracer)(tensor_to_funsor)


@quote.register(DeviceArray)
def _quote(x, indent, out):
    """
    Work around JAX's DeviceArray not supporting reproducible repr.
    """
    out.append(
        (indent, "np.array({}, dtype=np.{})".format(repr(x.copy().tolist()), x.dtype))
    )
