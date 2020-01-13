from collections import OrderedDict

import numpy as np
import torch
from multipledispatch import dispatch

import funsor
import funsor.ops as ops
from funsor.domains import bint
from funsor.terms import Funsor, Number, substitute


def align_tensor(new_inputs, x, expand=False):
    r"""
    Permute and add dims to a tensor to match desired ``new_inputs``.

    :param OrderedDict new_inputs: A target set of inputs.
    :param funsor.terms.Funsor x: A :class:`Tensor`, :class:`Array`, or
        :class:`~funsor.terms.Number` .
    :param bool expand: If False (default), set result size to 1 for any input
        of ``x`` not in ``new_inputs``; if True expand to ``new_inputs`` size.
    :return: a number or :class:`torch.Tensor` that can be broadcast to other
        tensors with inputs ``new_inputs``.
    :rtype: int or float or torch.Tensor or numpy.ndarray
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(x, (Number, funsor.torch.Tensor, funsor.numpy.Array))
    assert all(isinstance(d.dtype, int) for d in x.inputs.values())

    data = x.data
    if isinstance(x, Number):
        return data

    old_inputs = x.inputs
    if old_inputs == new_inputs:
        return data

    # Permute squashed input dims.
    x_keys = tuple(old_inputs)
    data = ops.permute(data, tuple(x_keys.index(k) for k in new_inputs if k in old_inputs) +
                       tuple(range(len(old_inputs), len(data.shape))))

    # Unsquash multivariate input dims by filling in ones.
    data = data.reshape(tuple(old_inputs[k].dtype if k in old_inputs else 1 for k in new_inputs) +
                        x.output.shape)

    # Optionally expand new dims.
    if expand:
        data = ops.expand(data, tuple(d.dtype for d in new_inputs.values()) + x.output.shape)
    return data


def align_tensors(*args, **kwargs):
    r"""
    Permute multiple tensors before applying a broadcasted op.

    This is mainly useful for implementing eager funsor operations.

    :param funsor.terms.Funsor \*args: Multiple :class:`Tensor` s or :class:`Array` s and
        :class:`~funsor.terms.Number` s.
    :param bool expand: Whether to expand input tensors. Defaults to False.
    :return: a pair ``(inputs, tensors)`` where tensors are all
        :class:`torch.Tensor` s that can be broadcast together to a single data
        with given ``inputs``.
    :rtype: tuple
    """
    expand = kwargs.pop('expand', False)
    assert not kwargs
    inputs = OrderedDict()
    for x in args:
        inputs.update(x.inputs)
    tensors = [align_tensor(inputs, x, expand=expand) for x in args]
    return inputs, tensors


def arange(prototype, name, *args, **kwargs):
    """
    Helper to create a named :func:`torch.arange` funsor.
    In some cases this can be replaced by a symbolic
    :class:`~funsor.terms.Slice` .

    :param prototype: either a torch.Tensor or a numpy.array
    :param str name: A variable name.
    :param int start:
    :param int stop:
    :param int step: Three args following :py:class:`slice` semantics.
    :param int dtype: An optional bounded integer type of this slice.
    :rtype: Tensor
    """
    start = 0
    step = 1
    dtype = None
    if len(args) == 1:
        stop = args[0]
        dtype = kwargs.pop("dtype", stop)
    elif len(args) == 2:
        start, stop = args
        dtype = kwargs.pop("dtype", stop)
    elif len(args) == 3:
        start, stop, step = args
        dtype = kwargs.pop("dtype", stop)
    elif len(args) == 4:
        start, stop, step, dtype = args
    else:
        raise ValueError
    if step <= 0:
        raise ValueError
    stop = min(dtype, max(start, stop))
    data = ops.new_arange(prototype, start, stop, step)
    inputs = OrderedDict([(name, bint(len(data)))])
    return Tensor(data, inputs, dtype=dtype)


def materialize(prototype, x):
    """
    Attempt to convert a Funsor to a :class:`~funsor.terms.Number` or
    :class:`Tensor` by substituting :func:`arange` s into its free variables.

    :param prototype: either a torch.Tensor or a numpy.array
    :arg Funsor x: A funsor.
    :rtype: Funsor
    """
    assert isinstance(x, Funsor)
    if isinstance(x, (Number, funsor.torch.Tensor, funsor.numpy.Array)):
        return x
    subs = []
    for name, domain in x.inputs.items():
        if isinstance(domain.dtype, int):
            subs.append((name, arange(prototype, name, domain.dtype)))
    subs = tuple(subs)
    return substitute(x, subs)


@dispatch(torch.Tensor, (type(None), tuple, OrderedDict), (int, str))
def _Tensor(x, inputs, dtype):
    return funsor.torch.Tensor(x, inputs, dtype)


@dispatch(np.ndarray, (type(None), tuple, OrderedDict), (int, str))
def _Tensor(x, inputs, dtype):
    return funsor.numpy.Array(x, inputs, dtype)


def Tensor(x, inputs=None, dtype="real"):
    return _Tensor(x, inputs, dtype)


def is_tensor(x):
    return isinstance(x, (funsor.torch.Tensor, funsor.numpy.Array))
