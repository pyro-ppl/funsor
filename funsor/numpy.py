from collections import OrderedDict

import numpy as np
from multipledispatch import dispatch

import funsor.ops as ops
from funsor.domains import Domain, bint, find_domain
from funsor.terms import Binary, Funsor, FunsorMeta, Number, eager, substitute, to_data, to_funsor


class ArrayMeta(FunsorMeta):
    """
    Wrapper to fill in default args and convert between OrderedDict and tuple.
    """
    def __call__(cls, data, inputs=None, dtype="real"):
        if inputs is None:
            inputs = tuple()
        elif isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        return super(ArrayMeta, cls).__call__(data, inputs, dtype)


class Array(Funsor, metaclass=ArrayMeta):
    """
    Funsor backed by a NumPy Array.

    This follows the :mod:`torch.distributions` convention of arranging
    named "batch" dimensions on the left and remaining "event" dimensions
    on the right. The output shape is determined by all remaining dims.
    For example::

        data = np.zeros((5,4,3,2))
        x = Array(data, OrderedDict([("i", bint(5)), ("j", bint(4))]))
        assert x.output == reals(3, 2)

    Operators like ``matmul`` and ``.sum()`` operate only on the output shape,
    and will not change the named inputs.

    :param np.ndarray data: A NumPy array.
    :param OrderedDict inputs: An optional mapping from input name (str) to
        datatype (:class:`~funsor.domains.Domain` ). Defaults to empty.
    :param dtype: optional output datatype. Defaults to "real".
    :type dtype: int or the string "real".
    """
    def __init__(self, data, inputs=None, dtype="real"):
        assert isinstance(data, np.ndarray) or np.isscalar(data)
        assert isinstance(inputs, tuple)
        assert all(isinstance(d.dtype, int) for k, d in inputs)
        inputs = OrderedDict(inputs)
        output = Domain(data.shape[len(inputs):], dtype)
        fresh = frozenset(inputs.keys())
        bound = frozenset()
        super(Array, self).__init__(inputs, output, fresh, bound)
        self.data = data

    def __repr__(self):
        if self.output != "real":
            return 'Array({}, {}, {})'.format(self.data, self.inputs, repr(self.dtype))
        elif self.inputs:
            return 'Array({}, {})'.format(self.data, self.inputs)
        else:
            return 'Array({})'.format(self.data)

    def __str__(self):
        if self.dtype != "real":
            return 'Array({}, {}, {})'.format(self.data, self.inputs, repr(self.dtype))
        elif self.inputs:
            return 'Array({}, {})'.format(self.data, self.inputs)
        else:
            return str(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data.item()

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self
        inputs = OrderedDict((name, self.inputs[name]) for name in names)
        inputs.update(self.inputs)

        if any(d.shape for d in self.inputs.values()):
            raise NotImplementedError("TODO: Implement align with vector indices.")
        old_dims = tuple(self.inputs)
        new_dims = tuple(inputs)
        data = np.transpose(self.data, (tuple(old_dims.index(d) for d in new_dims)))
        return Array(data, inputs, self.dtype)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = {k: materialize(v) for k, v in subs if k in self.inputs}
        if not subs:
            return self

        # Compute result shapes.
        inputs = OrderedDict()
        for k, domain in self.inputs.items():
            if k in subs:
                inputs.update(subs[k].inputs)
            else:
                inputs[k] = domain

        # Construct a dict with each input's positional dim,
        # counting from the right so as to support broadcasting.
        total_size = len(inputs) + len(self.output.shape)  # Assumes only scalar indices.
        new_dims = {}
        for k, domain in inputs.items():
            assert not domain.shape
            new_dims[k] = len(new_dims) - total_size

        # Use advanced indexing to construct a simultaneous substitution.
        index = []
        for k, domain in self.inputs.items():
            if k in subs:
                v = subs.get(k)
                if isinstance(v, Number):
                    index.append(int(v.data))
                else:
                    # Permute and expand v.data to end up at new_dims.
                    assert isinstance(v, Array)
                    v = v.align(tuple(k2 for k2 in inputs if k2 in v.inputs))
                    assert isinstance(v, Array)
                    v_shape = [1] * total_size
                    for k2, size in zip(v.inputs, v.data.shape):
                        v_shape[new_dims[k2]] = size
                    index.append(v.data.reshape(tuple(v_shape)))
            else:
                # Construct a [:] slice for this preserved input.
                offset_from_right = -1 - new_dims[k]
                index.append(np.arange(domain.dtype).reshape(
                    (-1,) + (1,) * offset_from_right))

        # Construct a [:] slice for the output.
        for i, size in enumerate(self.output.shape):
            offset_from_right = len(self.output.shape) - i - 1
            index.append(np.arange(size).reshape(
                (-1,) + (1,) * offset_from_right))

        data = self.data[tuple(index)]
        return Array(data, inputs, self.dtype)


@ops.TensorOp.register(np.ndarray, (type(None), tuple, OrderedDict), str)
def _Tensor(x, inputs, dtype):
    return Array(x, inputs, dtype)


@dispatch(np.ndarray)
def to_funsor(x):
    return Array(x)


@dispatch(np.ndarray, Domain)
def to_funsor(x, output):
    result = Array(x, dtype=output.dtype)
    if result.output != output:
        raise ValueError("Invalid shape: expected {}, actual {}"
                         .format(output.shape, result.output.shape))
    return result


@ops.align_tensor_op.register(OrderedDict, Array, bool)
def align_array(new_inputs, x, expand=False):
    r"""
    Permute and expand an array to match desired ``new_inputs``.

    :param OrderedDict new_inputs: A target set of inputs.
    :param funsor.terms.Funsor x: A :class:`Array` s or
        or :class:`~funsor.terms.Number` .
    :param bool expand: If False (default), set result size to 1 for any input
        of ``x`` not in ``new_inputs``; if True expand to ``new_inputs`` size.
    :return: a number or :class:`numpy.ndarray` that can be broadcast to other
        array with inputs ``new_inputs``.
    :rtype: tuple
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(x, (Number, Array))
    assert all(isinstance(d.dtype, int) for d in x.inputs.values())

    data = x.data
    if isinstance(x, Number):
        return data

    old_inputs = x.inputs
    if old_inputs == new_inputs:
        return data

    # Permute squashed input dims.
    x_keys = tuple(old_inputs)
    data = np.transpose(data, (tuple(x_keys.index(k) for k in new_inputs if k in old_inputs) +
                               tuple(range(len(old_inputs), data.ndim))))

    # Unsquash multivariate input dims by filling in ones.
    data = np.reshape(data, tuple(old_inputs[k].dtype if k in old_inputs else 1 for k in new_inputs) +
                      x.output.shape)

    # Optionally expand new dims.
    if expand:
        data = np.broadcast_to(data, tuple(d.dtype for d in new_inputs.values()) + x.output.shape)
    return data


def align_arrays(*args, **kwargs):
    r"""
    Permute multiple arrays before applying a broadcasted op.

    This is mainly useful for implementing eager funsor operations.

    :param funsor.terms.Funsor \*args: Multiple :class:`Array` s and
        :class:`~funsor.terms.Number` s.
    :param bool expand: Whether to expand input tensors. Defaults to False.
    :return: a pair ``(inputs, arrays)`` where arrayss are all
        :class:`numpy.ndarray` s that can be broadcast together to a single data
        with given ``inputs``.
    :rtype: tuple
    """
    expand = kwargs.pop('expand', False)
    assert not kwargs
    inputs = OrderedDict()
    for x in args:
        inputs.update(x.inputs)
    arrays = [align_array(inputs, x, expand=expand) for x in args]
    return inputs, arrays


@to_data.register(Array)
def _to_data_array(x):
    if x.inputs:
        raise ValueError(f"cannot convert Array to data due to lazy inputs: {set(x.inputs)}")
    return x.data


@eager.register(Binary, object, Array, Number)
def eager_binary_array_number(op, lhs, rhs):
    if op is ops.getitem:
        # Shift by that Funsor is using for inputs.
        index = [slice(None)] * len(lhs.inputs)
        index.append(rhs.data)
        index = tuple(index)
        data = lhs.data[index]
    else:
        data = op(lhs.data, rhs.data)
    return Array(data, lhs.inputs, lhs.dtype)


@eager.register(Binary, object, Number, Array)
def eager_binary_number_array(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    return Array(data, rhs.inputs, rhs.dtype)


@eager.register(Binary, object, Array, Array)
def eager_binary_array_array(op, lhs, rhs):
    # Compute inputs and outputs.
    dtype = find_domain(op, lhs.output, rhs.output).dtype
    if lhs.inputs == rhs.inputs:
        inputs = lhs.inputs
        lhs_data, rhs_data = lhs.data, rhs.data
    else:
        inputs, (lhs_data, rhs_data) = align_arrays(lhs, rhs)

    if op is ops.getitem:
        # getitem has special shape semantics.
        if rhs.output.shape:
            raise NotImplementedError('TODO support vector indexing')
        assert lhs.output.shape == (rhs.dtype,)
        index = [np.arange(size).reshape((-1,) + (1,) * (lhs_data.ndim - pos - 2))
                 for pos, size in enumerate(lhs_data.shape)]
        index[-1] = rhs_data
        data = lhs_data[tuple(index)]
    else:
        data = op(lhs_data, rhs_data)

    return Array(data, inputs, dtype)


def arange(name, size):
    """
    Helper to create a named :func:`numpy.arange` funsor.

    :param str name: A variable name.
    :param int size: A size.
    :rtype: Array
    """
    data = np.arange(size)
    inputs = OrderedDict([(name, bint(size))])
    return Array(data, inputs, dtype=size)


def materialize(x):
    """
    Attempt to convert a Funsor to a :class:`~funsor.terms.Number` or
    :class:`numpy.ndarray` by substituting :func:`arange` s into its free variables.
    """
    assert isinstance(x, Funsor)
    if isinstance(x, (Number, Array)):
        return x
    subs = []
    for name, domain in x.inputs.items():
        if not isinstance(domain.dtype, int):
            raise ValueError('materialize() requires integer free variables but found '
                             '"{}" of domain {}'.format(name, domain))
        assert not domain.shape
        subs.append((name, arange(name, domain.dtype)))
    subs = tuple(subs)
    return substitute(x, subs)


@ops.materialize.register(np.ndarray, Funsor)
def _materialize(prototype, x):
    return materialize(x)


################################################################################
# Register Ops
################################################################################


@ops.abs.register(np.ndarray)
def _abs(x):
    return abs(x)


@ops.sigmoid.register(np.ndarray)
def _sigmoid(x):
    try:
        from scipy.special import expit
        return expit(x)
    except ImportError:
        return 1 / (1 + np.exp(-x))


@ops.sqrt.register(np.ndarray)
def _sqrt(x):
    return np.sqrt(x)


@ops.exp.register(np.ndarray)
def _exp(x):
    return np.exp(x)


@ops.log.register(np.ndarray)
def _log(x):
    return np.log(x)


@ops.log1p.register(np.ndarray)
def _log1p(x):
    return np.log1p(x)


@ops.min.register(np.ndarray, np.ndarray)
def _min(x, y):
    return np.minimum(x, y)


# TODO: replace (int, float) by object
@ops.min.register((int, float), np.ndarray)
def _min(x, y):
    return np.clip(y, a_max=x)


@ops.min.register(np.ndarray, (int, float))
def _min(x, y):
    return np.clip(x, a_max=y)


@ops.max.register(np.ndarray, np.ndarray)
def _max(x, y):
    return np.maximum(x, y)


@ops.max.register((int, float), np.ndarray)
def _max(x, y):
    return np.clip(y, a_min=x)


@ops.max.register(np.ndarray, (int, float))
def _max(x, y):
    return np.clip(x, a_min=y)


@ops.reciprocal.register(np.ndarray)
def _reciprocal(x):
    result = np.clip(np.reciprocal(x), a_max=np.finfo(x.dtype).max)
    return result


@ops.safesub.register((int, float), np.ndarray)
def _safesub(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x + np.clip(-y, a_max=finfo)


@ops.safediv.register((int, float), np.ndarray)
def _safediv(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x * np.clip(np.reciprocal(y), a_max=finfo)


@ops.cholesky.register(np.ndarray)
def _cholesky(x):
    """
    Like :func:`numpy.linalg.cholesky` but uses sqrt for scalar matrices.
    """
    if x.shape[-1] == 1:
        return np.sqrt(x)
    return np.linalg.cholesky(x)


@ops.cholesky_inverse.register(np.ndarray)
def _cholesky_inverse(x):
    """
    Like :func:`torch.cholesky_inverse` but supports batching and gradients.
    """
    from scipy.linalg import cho_solve

    return cho_solve((x, False), np.eye(x.shape[-1]))


@ops.triangular_solve_op.register(np.ndarray, np.ndarray, bool, bool)
def _triangular_solve(x, y, upper, transpose):
    from scipy.linalg import solve_triangular

    # TODO: remove this logic when using JAX
    # work around the issue of scipy which does not support batched input
    batch_shape = np.broadcast(x[..., 0, 0], y[..., 0, 0]).shape
    xs = np.broadcast_to(x, batch_shape + x.shape[-2:]).reshape((-1,) + x.shape[-2:])
    ys = np.broadcast_to(y, batch_shape + y.shape[-2:]).reshape((-1,) + y.shape[-2:])
    ans = [solve_triangular(y, x, trans=int(transpose), lower=not upper)
           for (x, y) in zip(xs, ys)]
    ans = np.stack(ans)
    return ans.reshape(batch_shape + ans.shape[-2:])


@ops.diagonal.register(np.ndarray, int, int)
def _diagonal(x, dim1, dim2):
    return np.diagonal(x, axis1=dim1, axis2=dim2)


@ops.cat_op.register(int, [np.ndarray])
def _cat(dim, *x):
    return np.concatenate(x, axis=dim)


@ops.new_zeros.register(np.ndarray, tuple)
def _new_zeros(x, shape):
    return np.zeros(shape, dtype=x.dtype)


@ops.new_eye.register(np.ndarray, tuple)
def _new_eye(x, shape):
    return np.broadcast_to(np.eye(shape[-1]), shape + (-1,))


@ops.unsqueeze.register(np.ndarray, int)
def _unsqueeze(x, dim):
    return np.expand_dims(x, dim)


@ops.expand.register(np.ndarray, tuple)
def _expand(x, shape):
    return np.broadcast_to(x, shape)


@ops.transpose.register(np.ndarray, int, int)
def _transpose(x, dim0, dim1):
    return np.swapaxes(x, dim0, dim1)
