# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from funsor import ops
from funsor.domains import reals
from funsor.tensor import function, register_bijection


def spline_call(x, arg1, arg2, arg3):
    # you can use basic arithmetic
    u = arg1 + arg2
    # or use ops.* for fancy math
    v = ops.exp(arg3)
    # if you use a torch.nn or something backend-specific, be sure to wrap it
    # in a funsor.tensor.function.
    return x + u + v  # or something


def spline_inv(y, arg1, arg2, arg3):
    # whatever
    u = arg1 + arg2
    v = ops.exp(arg3)
    return y - u - v


def spline_log_abs_det_jacobian(x, y, arg1, arg2, arg3):
    return 0.


register_bijection(spline_call, spline_inv, spline_log_abs_det_jacobian)

# TODO Find a pattern that allows varying size.
spline = function(reals(), reals(), reals(), reals(), reals())(spline_call)

__all__ = ["spline"]
