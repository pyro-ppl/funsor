# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict, defaultdict
from functools import reduce

import numpy as np

import funsor
import funsor.ops as ops
from funsor.affine import affine_inputs, extract_affine, is_affine
from funsor.delta import Delta
from funsor.domains import Real, Reals
from funsor.ops import AddOp, NegOp, SubOp
from funsor.tensor import Tensor, align_tensor, align_tensors
from funsor.terms import (
    Align,
    Binary,
    Funsor,
    FunsorMeta,
    Number,
    Slice,
    Subs,
    Unary,
    Variable,
    eager,
    reflect,
)
from funsor.util import broadcast_shape, get_backend, get_tracing_state, lazy_property


def _compress_rank(white_vec, prec_sqrt):
    """
    Compress a wide representation ``(white_vec, prec_sqrt)`` while preserving
    the quadratic function ``||x @ prec_sqrt - white_vec||^2``.
    """
    dim, rank = prec_sqrt.shape[-2:]
    if rank <= dim:
        return white_vec, prec_sqrt, None
    old_norm2 = _norm2(white_vec)
    info_vec_ = prec_sqrt @ white_vec[..., None]
    precision = prec_sqrt @ ops.transpose(prec_sqrt, -1, -2)
    prec_sqrt = ops.cholesky(precision)
    white_vec = ops.triangular_solve(
        info_vec_, prec_sqrt  # , upper=True, transpose=True
    )[..., 0]
    new_norm2 = _norm2(white_vec)
    shift = 0.5 * (new_norm2 - old_norm2)
    return white_vec, prec_sqrt, shift


def _log_det_tri(x):
    return ops.log(ops.diagonal(x, -1, -2)).sum(-1)


def _vv(vec1, vec2):
    """
    Computes the inner product ``< vec1 | vec 2 >``.
    """
    return (vec1[..., None, :] @ vec2[..., None])[..., 0, 0]


def _norm2(vec):
    return _vv(vec, vec)


def _mv(mat, vec):
    return (mat @ vec[..., None])[..., 0]


def _vm(vec, mat):
    return (vec[..., None, :] @ mat)[..., 0, :]


def _mmtv(m1, m2, v):
    return (m1 @ (ops.transpose(m2, -1, -2) @ v[..., None]))[..., 0]


def _trace_mm(x, y):
    """
    Computes ``trace(x.T @ y)``.
    """
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return (x * y).sum((-1, -2))


def _compute_offsets(inputs):
    """
    Compute offsets of real inputs into the concatenated Gaussian dims.
    This ignores all int inputs.

    :param OrderedDict inputs: A schema mapping variable name to domain.
    :return: a pair ``(offsets, total)``, where ``offsets`` is an OrderedDict
        mapping input name to integer offset, and ``total`` is the total event
        size.
    :rtype: tuple
    """
    assert isinstance(inputs, OrderedDict)
    offsets = OrderedDict()
    total = 0
    for key, domain in inputs.items():
        if domain.dtype == "real":
            offsets[key] = total
            total += domain.num_elements
    return offsets, total


def _find_intervals(intervals, end):
    """
    Finds a complete set of intervals partitioning [0, end), given a partial
    set of non-overlapping intervals.
    """
    cuts = list(sorted({0, end}.union(*intervals)))
    return list(zip(cuts[:-1], cuts[1:]))


def _parse_slices(index, value):
    if not isinstance(index, tuple):
        index = (index,)
    if index[0] is Ellipsis:
        index = index[1:]
    start_stops = []
    for pos, i in reversed(list(enumerate(index))):
        if isinstance(i, slice):
            start_stops.append((i.start, i.stop))
        elif isinstance(i, int):
            start_stops.append((i, i + 1))
            value = ops.unsqueeze(value, pos - len(index))
        else:
            raise ValueError("invalid index: {}".format(i))
    start_stops.reverse()
    return start_stops, value


class BlockVector(object):
    """
    Jit-compatible helper to build blockwise vectors.
    Syntax is similar to :func:`torch.zeros` ::

        x = BlockVector((100, 20))
        x[..., 0:4] = x1
        x[..., 6:10] = x2
        x = x.as_tensor()
        assert x.shape == (100, 20)
    """

    def __init__(self, shape):
        self.shape = shape
        self.parts = {}

    def __setitem__(self, index, value):
        (i,), value = _parse_slices(index, value)
        self.parts[i] = value

    def as_tensor(self):
        # Fill gaps with zeros.
        prototype = next(iter(self.parts.values()))
        for i in _find_intervals(self.parts.keys(), self.shape[-1]):
            if i not in self.parts:
                self.parts[i] = ops.new_zeros(
                    prototype, self.shape[:-1] + (i[1] - i[0],)
                )

        # Concatenate parts.
        parts = [v for k, v in sorted(self.parts.items())]
        result = ops.cat(parts, -1)
        if not get_tracing_state():
            assert result.shape == self.shape
        return result


class BlockMatrix(object):
    """
    Jit-compatible helper to build blockwise matrices.
    Syntax is similar to :func:`torch.zeros` ::

        x = BlockMatrix((100, 20, 20))
        x[..., 0:4, 0:4] = x11
        x[..., 0:4, 6:10] = x12
        x[..., 6:10, 0:4] = x12.transpose(-1, -2)
        x[..., 6:10, 6:10] = x22
        x = x.as_tensor()
        assert x.shape == (100, 20, 20)
    """

    def __init__(self, shape):
        self.shape = shape
        self.parts = defaultdict(dict)

    def __setitem__(self, index, value):
        (i, j), value = _parse_slices(index, value)
        self.parts[i][j] = value

    def as_tensor(self):
        # Fill gaps with zeros.
        arbitrary_row = next(iter(self.parts.values()))
        prototype = next(iter(arbitrary_row.values()))
        js = set().union(*(part.keys() for part in self.parts.values()))
        rows = _find_intervals(self.parts.keys(), self.shape[-2])
        cols = _find_intervals(js, self.shape[-1])
        for i in rows:
            for j in cols:
                if j not in self.parts[i]:
                    shape = self.shape[:-2] + (i[1] - i[0], j[1] - j[0])
                    self.parts[i][j] = ops.new_zeros(prototype, shape)

        # Concatenate parts.
        # TODO This could be optimized into a single .reshape().cat().reshape() if
        #   all inputs are contiguous, thereby saving a memcopy.
        columns = {
            i: ops.cat([v for j, v in sorted(part.items())], -1)
            for i, part in self.parts.items()
        }
        result = ops.cat([v for i, v in sorted(columns.items())], -2)
        if not get_tracing_state():
            assert result.shape == self.shape
        return result


def align_gaussian(new_inputs, old, expand=False):
    """
    Align data of a Gaussian distribution to a new ``inputs`` shape.
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(old, Gaussian)
    white_vec = old.white_vec
    prec_sqrt = old.prec_sqrt

    # Align int inputs.
    # Since these are are managed as in Tensor, we can defer to align_tensor().
    new_ints = OrderedDict((k, d) for k, d in new_inputs.items() if d.dtype != "real")
    old_ints = OrderedDict((k, d) for k, d in old.inputs.items() if d.dtype != "real")
    if new_ints != old_ints:
        white_vec = align_tensor(new_ints, Tensor(white_vec, old_ints), expand=expand)
        prec_sqrt = align_tensor(new_ints, Tensor(prec_sqrt, old_ints), expand=expand)

    # Align real inputs, which are all concatenated in the rightmost dims.
    new_offsets, new_dim = _compute_offsets(new_inputs)
    old_offsets, old_dim = _compute_offsets(old.inputs)
    assert prec_sqrt.shape[-2:-1] == (old_dim,)
    if new_offsets != old_offsets:
        old_prec_sqrt = ops.transpose(prec_sqrt, -1, -2)
        prec_sqrt = BlockVector(old_prec_sqrt.shape[:-1] + (new_dim,))
        for k, new_offset in new_offsets.items():
            if k not in old_offsets:
                continue
            offset = old_offsets[k]
            num_elements = old.inputs[k].num_elements
            old_slice = slice(offset, offset + num_elements)
            new_slice = slice(new_offset, new_offset + num_elements)
            prec_sqrt[..., new_slice] = old_prec_sqrt[..., old_slice]
        prec_sqrt = prec_sqrt.as_tensor()
        prec_sqrt = ops.transpose(prec_sqrt, -1, -2)

    return white_vec, prec_sqrt


class GaussianMeta(FunsorMeta):
    """
    Wrapper to convert from external to internal compressed representation.

    This may return either a Gaussian or a Gaussian + Tensor, where the Tensor
    represents byproducts of compression.
    """

    def __call__(
        cls,
        white_vec=None,
        prec_sqrt=None,
        inputs=None,
        negate=None,
        *,
        mean=None,
        precision=None,
        scale_tril=None,
        covariance=None,
    ):
        # Convert inputs.
        assert inputs is not None
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        assert isinstance(inputs, tuple)

        # Convert prec_sqrt.
        if prec_sqrt is not None:
            pass
        elif precision is not None:
            prec_sqrt = ops.cholesky(precision)
        elif scale_tril is not None:
            prec_sqrt = ops.triangular_inv(scale_tril, upper=True, transpose=True)
        elif covariance is not None:
            scale_tril = ops.cholesky(covariance)
            prec_sqrt = ops.triangular_inv(scale_tril, upper=True, transpose=True)
        else:
            raise ValueError(
                "At least one of prec_sqrt, precision, scale_tril, or covariance "
                "must be specified"
            )

        # Convert white_vec.
        if white_vec is not None:
            pass
        elif mean is not None:
            white_vec = (mean[..., None, :] @ prec_sqrt)[..., 0, :]
        else:
            raise ValueError("At least one of white_vec or mean must be specified")

        # Compress wide representations.
        white_vec, prec_sqrt, shift = _compress_rank(white_vec, prec_sqrt)

        # Create a Gaussian.
        assert isinstance(negate, bool)
        result = super().__call__(white_vec, prec_sqrt, inputs, negate)

        # Add compression byproducts.
        if shift is not None:
            if negate:
                shift = -shift
            int_inputs = OrderedDict(
                (k, v) for k, v in inputs.items() if v.dtype != "real"
            )
            result += Tensor(shift, int_inputs)

        return result


class Gaussian(Funsor, metaclass=GaussianMeta):
    r"""
    Funsor representing a batched joint Gaussian distribution as a log-density
    function.

    Mathematically, a Gaussian represents the quadratic log density function::

        f(x) = -0.5 * || x @ prec_sqrt - white_vec ||^2
             = -0.5 * < x @ prec_sqrt - white_vec | x @ prec_sqrt - white_vec >
             = -0.5 * < x | prec_sqrt @ prec_sqrt.T | x>
               + < x | prec_sqrt | white_vec > - 0.5 ||white_vec||^2

    Note that :class:`Gaussian` s are not normalized probability distributions,
    rather they are canonicalized to evaluate to zero log density at their
    extremum: ``f(prec_sqrt \ white_vec) = 0``. This canonical form is useful
    in combination with the square root information filter (SRIF)
    representation because it allows :class:`Gaussian` s with incomplete
    information, i.e. with zero eigenvalues in the precision matrix. These
    incomplete distributions arise when making low-dimensional observations on
    higher-dimensional hidden state.

    Not only are Gaussians non-normalized, but they may be rank deficient and
    non-normalizable, in which case sampling and marginalization are not
    supported. See the :meth:`is_full_rank` property.

    :param torch.Tensor white_vec: An batched white noise vector, where
        ``white_vec = prec_sqrt.T @ mean``. Alternatively you can specify the
        kwarg ``mean``, which will be converted to ``white_vec``.
    :param torch.Tensor prec_sqrt: A batched square root of the positive
        semidefinite precision matrix. This need not be square, and typically
        has shape ``prec_sqrt.shape == white_vec.shape[:-1] + (dim, rank)``,
        where ``dim`` is the total flattened size of real inputs and
        ``rank = white_vec.shape[-1]``.  Alternatively you can specify one of
        the kwargs ``precision``, ``covariance``, or ``scale_tril``, which will
        be converted to ``prec_sqrt``.
    :param OrderedDict inputs: Mapping from name to
        :class:`~funsor.domains.Domain` .
    :param bool negate: If false this represents a concave density 🙁. If true,
        this represents a convex density 🙂. Convex densities are not
        normalizable and do not support marginalization or sampling. Defaults
        to False (concave).
    """

    def __init__(self, white_vec, prec_sqrt, inputs, negate):
        assert ops.is_numeric_array(white_vec) and ops.is_numeric_array(prec_sqrt)
        assert isinstance(inputs, tuple)
        inputs = OrderedDict(inputs)

        # Compute total dimension of all real inputs.
        dim = sum(d.num_elements for d in inputs.values() if d.dtype == "real")
        if not get_tracing_state():
            assert dim
            assert len(prec_sqrt.shape) >= 2 and prec_sqrt.shape[-2] == dim
            rank = prec_sqrt.shape[-1]
            assert len(white_vec.shape) >= 1 and white_vec.shape[-1] == rank
            assert rank <= dim

        # Compute total shape of all Bint inputs.
        batch_shape = tuple(
            d.dtype for d in inputs.values() if isinstance(d.dtype, int)
        )
        if not get_tracing_state():
            assert prec_sqrt.shape[:-2] == batch_shape
            assert white_vec.shape[:-1] == batch_shape

        output = Real
        fresh = frozenset(inputs.keys())
        bound = {}
        super().__init__(inputs, output, fresh, bound)
        self.white_vec = white_vec
        self.prec_sqrt = prec_sqrt
        self.negate = negate
        self.batch_shape = batch_shape
        self.event_shape = (dim,)

    def __repr__(self):
        return "Gaussian(..., ({}))".format(
            " ".join("({}, {}),".format(*kv) for kv in self.inputs.items())
        )

    @property
    def is_full_rank(self):
        dim, rank = self.prec_sqrt.shape[-2:]
        return rank == dim

    # TODO Consider weak-memoizing these so they persist through alpha conversion.
    # https://github.com/pyro-ppl/pyro/blob/ac3c588/pyro/distributions/coalescent.py#L412
    @lazy_property
    def _precision(self):
        return self.prec_sqrt @ ops.transpose(self.prec_sqrt, -1, -2)

    @lazy_property
    def _precision_chol(self):
        return ops.cholesky(self._precision)

    @lazy_property
    def _covariance(self):
        assert self.is_full_rank
        return ops.cholesky_inverse(self._precision_chol)

    @lazy_property
    def _mean(self):
        return ops.triangular_solve(
            self.white_vec[..., None], self._precision_chol, transpose=True
        )[..., 0]

    @lazy_property
    def _log_normalizer(self):
        dim = self.prec_sqrt.shape[-2]
        log_det_term = _log_det_tri(self._precision_chol)
        return 0.5 * dim * math.log(2 * math.pi) - log_det_term

    @lazy_property
    def log_normalizer(self):
        inputs = OrderedDict(
            (k, v) for k, v in self.inputs.items() if v.dtype != "real"
        )
        return Tensor(self._log_normalizer, inputs)

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self

        inputs = OrderedDict((name, self.inputs[name]) for name in names)
        inputs.update(self.inputs)
        white_vec, prec_sqrt = align_gaussian(inputs, self)
        return Gaussian(white_vec, prec_sqrt, inputs, self.negate)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        prototype = Tensor(self.white_vec)
        subs = tuple(
            (k, v if isinstance(v, (Variable, Slice)) else prototype.materialize(v))
            for k, v in subs
            if k in self.inputs
        )
        if not subs:
            return self

        # Constants and Affine funsors are eagerly substituted;
        # everything else is lazily substituted.
        lazy_subs = tuple(
            (k, v)
            for k, v in subs
            if not isinstance(v, (Number, Tensor, Variable, Slice))
            and not (is_affine(v) and affine_inputs(v))
        )
        var_subs = tuple((k, v) for k, v in subs if isinstance(v, Variable))
        int_subs = tuple(
            (k, v)
            for k, v in subs
            if isinstance(v, (Number, Tensor, Slice))
            if v.dtype != "real"
        )
        real_subs = tuple(
            (k, v)
            for k, v in subs
            if isinstance(v, (Number, Tensor))
            if v.dtype == "real"
        )
        affine_subs = tuple(
            (k, v)
            for k, v in subs
            if is_affine(v) and affine_inputs(v) and not isinstance(v, Variable)
        )
        if var_subs:
            return self._eager_subs_var(
                var_subs, int_subs + real_subs + affine_subs + lazy_subs
            )
        if int_subs:
            return self._eager_subs_int(int_subs, real_subs + affine_subs + lazy_subs)
        if real_subs:
            return self._eager_subs_real(real_subs, affine_subs + lazy_subs)
        if affine_subs:
            return self._eager_subs_affine(affine_subs, lazy_subs)
        return reflect.interpret(Subs, self, lazy_subs)

    def _eager_subs_var(self, subs, remaining_subs):
        # Perform variable substitution, i.e. renaming of inputs.
        rename = {k: v.name for k, v in subs}
        inputs = OrderedDict((rename.get(k, k), d) for k, d in self.inputs.items())
        if len(inputs) != len(self.inputs):
            raise ValueError("Variable substitution name conflict")
        var_result = Gaussian(self.white_vec, self.prec_sqrt, inputs, self.negate)
        return Subs(var_result, remaining_subs) if remaining_subs else var_result

    def _eager_subs_int(self, subs, remaining_subs):
        # Perform integer substitution, i.e. slicing into a batch.
        int_inputs = OrderedDict(
            (k, d) for k, d in self.inputs.items() if d.dtype != "real"
        )
        real_inputs = OrderedDict(
            (k, d) for k, d in self.inputs.items() if d.dtype == "real"
        )
        tensors = [self.white_vec, self.prec_sqrt]
        funsors = [Subs(Tensor(x, int_inputs), subs) for x in tensors]
        inputs = funsors[0].inputs.copy()
        inputs.update(real_inputs)
        int_result = Gaussian(funsors[0].data, funsors[1].data, inputs, self.negate)
        return Subs(int_result, remaining_subs) if remaining_subs else int_result

    def _eager_subs_real(self, subs, remaining_subs):
        # Broadcast all component tensors.
        subs = OrderedDict(subs)
        int_inputs = OrderedDict(
            (k, d) for k, d in self.inputs.items() if d.dtype != "real"
        )
        tensors = [
            Tensor(self.white_vec, int_inputs),
            Tensor(self.prec_sqrt, int_inputs),
        ]
        tensors.extend(subs.values())
        int_inputs, tensors = align_tensors(*tensors)
        batch_dim = len(tensors[0].shape) - 1
        batch_shape = broadcast_shape(*(x.shape[:batch_dim] for x in tensors))
        (white_vec, prec_sqrt), values = tensors[:2], tensors[2:]
        offsets, event_size = _compute_offsets(self.inputs)
        slices = [
            (k, slice(offset, offset + self.inputs[k].num_elements))
            for k, offset in offsets.items()
        ]

        # Expand all substituted values.
        values = OrderedDict(zip(subs, values))
        for k, value in values.items():
            value = value.reshape(value.shape[:batch_dim] + (-1,))
            if not get_tracing_state():
                assert value.shape[-1] == self.inputs[k].num_elements
            values[k] = ops.expand(value, batch_shape + value.shape[-1:])

        # Try to perform a complete substitution of all real variables,
        # resulting in a Tensor.
        if all(k in subs for k, d in self.inputs.items() if d.dtype == "real"):
            # Form the concatenated value.
            value = BlockVector(batch_shape + (event_size,))
            for k, i in slices:
                if k in values:
                    value[..., i] = values[k]
            value = value.as_tensor()

            # Evaluate the non-normalized log density.
            result = _norm2(_vm(value, prec_sqrt) - white_vec)
            result = (0.5 if self.negate else -0.5)

            result = Tensor(result, int_inputs)
            assert result.output == Real
            return Subs(result, remaining_subs) if remaining_subs else result

        # Perform a partial substution of a subset of real variables, resulting
        # in a Joint. We split real inputs into two sets: a for the preserved
        # and b for the substituted.
        #   f(x) = 0.5 * ||x @ prec_sqrt - white_vec||^2
        #        = 0.5 * ||x_a @ prec_sqrt_a + x_b @ prec_sqrt_b - white_vec||^2
        #        = 0.5 * ||x_a @ prec_sqrt_a - white_vec_a||^2
        # whence
        #   white_vec_a = white_vec - x_b @ prec_sqrt_b
        b = frozenset(k for k, v in subs.items())
        a = frozenset(
            k for k, d in self.inputs.items() if d.dtype == "real" and k not in b
        )
        prec_sqrt_a = ops.cat([prec_sqrt[..., i, :] for k, i in slices if k in a], -2)
        prec_sqrt_b = ops.cat([prec_sqrt[..., i, :] for k, i in slices if k in b], -2)
        value_b = ops.cat([values[k] for k, i in slices if k in b], -1)
        white_vec_a = white_vec - _vm(value_b, prec_sqrt_b)
        prec_sqrt_a = ops.expand(prec_sqrt_a, white_vec_a.shape[:-1] + (-1, -1))
        inputs = int_inputs.copy()
        for k, d in self.inputs.items():
            if k not in subs:
                inputs[k] = d
        result = Gaussian(white_vec_a, prec_sqrt_a, inputs, self.negate)
        return Subs(result, remaining_subs) if remaining_subs else result

    def _eager_subs_affine(self, subs, remaining_subs):
        # Extract an affine representation.
        affine = OrderedDict()
        for k, v in subs:
            const, coeffs = extract_affine(v)
            if isinstance(const, Tensor) and all(
                isinstance(coeff, Tensor) for coeff, _ in coeffs.values()
            ):
                affine[k] = const, coeffs
            else:
                remaining_subs += ((k, v),)
        if not affine:
            return reflect.interpret(Subs, self, remaining_subs)

        # Align integer dimensions.
        old_int_inputs = OrderedDict(
            (k, v) for k, v in self.inputs.items() if v.dtype != "real"
        )
        tensors = [
            Tensor(self.white_vec, old_int_inputs),
            Tensor(self.prec_sqrt, old_int_inputs),
        ]
        for const, coeffs in affine.values():
            tensors.append(const)
            tensors.extend(coeff for coeff, _ in coeffs.values())
        new_int_inputs, tensors = align_tensors(*tensors, expand=True)
        tensors = (Tensor(x, new_int_inputs) for x in tensors)
        white_vec = next(tensors).data
        prec_sqrt = next(tensors).data
        for old_k, (const, coeffs) in affine.items():
            const = next(tensors)
            for new_k, (coeff, eqn) in coeffs.items():
                coeff = next(tensors)
                coeffs[new_k] = coeff, eqn
            affine[old_k] = const, coeffs
        batch_shape = white_vec.shape[:-1]

        # Align real dimensions.
        old_real_inputs = OrderedDict(
            (k, v) for k, v in self.inputs.items() if v.dtype == "real"
        )
        new_real_inputs = old_real_inputs.copy()
        for old_k, (const, coeffs) in affine.items():
            del new_real_inputs[old_k]
            for new_k, (coeff, eqn) in coeffs.items():
                new_shape = coeff.shape[: len(eqn.split("->")[0].split(",")[1])]
                new_real_inputs[new_k] = Reals[new_shape]
        old_offsets, old_dim = _compute_offsets(old_real_inputs)
        new_offsets, new_dim = _compute_offsets(new_real_inputs)
        new_inputs = new_int_inputs.copy()
        new_inputs.update(new_real_inputs)

        # Construct a blockwise affine representation of the substitution.
        subs_vector = BlockVector(batch_shape + (old_dim,))
        subs_matrix = BlockMatrix(batch_shape + (new_dim, old_dim))
        for old_k, old_offset in old_offsets.items():
            old_size = old_real_inputs[old_k].num_elements
            old_slice = slice(old_offset, old_offset + old_size)
            if old_k in new_real_inputs:
                new_offset = new_offsets[old_k]
                new_slice = slice(new_offset, new_offset + old_size)
                subs_matrix[..., new_slice, old_slice] = ops.new_eye(
                    self.white_vec, batch_shape + (old_size,)
                )
                continue
            const, coeffs = affine[old_k]
            old_shape = old_real_inputs[old_k].shape
            assert const.data.shape == batch_shape + old_shape
            subs_vector[..., old_slice] = const.data.reshape(batch_shape + (old_size,))
            for new_k, new_offset in new_offsets.items():
                if new_k in coeffs:
                    coeff, eqn = coeffs[new_k]
                    new_size = new_real_inputs[new_k].num_elements
                    new_slice = slice(new_offset, new_offset + new_size)
                    assert coeff.shape == new_real_inputs[new_k].shape + old_shape
                    subs_matrix[..., new_slice, old_slice] = coeff.data.reshape(
                        batch_shape + (new_size, old_size)
                    )
        subs_vector = subs_vector.as_tensor()
        subs_matrix = subs_matrix.as_tensor()

        # Construct the new funsor. Suppose the old Gaussian funsor g has density
        #   g(x) = 1/2 || x S - w ||^2
        # Now define a new funsor f by substituting x = A y + b:
        #   f(y) = g(y A + b)
        #        = 1/2 || (y A + b) S - w ||^2
        #        = 1/2 || y A S - w + b S ||^2
        #        =: 1/2 || y S' - w' ||^2
        # where  S' = A S  and  w' = w - b S  parametrize a new Gaussian.
        white_vec = white_vec - _vm(subs_vector, prec_sqrt)
        prec_sqrt = subs_matrix @ prec_sqrt
        result = Gaussian(white_vec, prec_sqrt, new_inputs, self.negate)
        return Subs(result, remaining_subs) if remaining_subs else result

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if op is ops.logaddexp:
            # Marginalize out real variables, but keep mixtures lazy.
            assert not self.negate
            assert all(v in self.inputs for v in reduced_vars)
            real_vars = frozenset(
                k for k, d in self.inputs.items() if d.dtype == "real"
            )
            reduced_reals = reduced_vars & real_vars
            reduced_ints = reduced_vars - real_vars
            if not reduced_reals:
                return None  # defer to default implementation

            inputs = OrderedDict(
                (k, d) for k, d in self.inputs.items() if k not in reduced_reals
            )
            if reduced_reals == real_vars:
                return self.log_normalizer.reduce(ops.logaddexp, reduced_ints)

            int_inputs = OrderedDict(
                (k, v) for k, v in inputs.items() if v.dtype != "real"
            )
            offsets, _ = _compute_offsets(self.inputs)

            a = []
            b = []
            for key, domain in self.inputs.items():
                if domain.dtype == "real":
                    block = ops.new_arange(
                        self.info_vec,
                        offsets[key],
                        offsets[key] + domain.num_elements,
                        1,
                    )
                    (b if key in reduced_vars else a).append(block)
            a = ops.cat(a, -1)
            b = ops.cat(b, -1)
            ###############################################################
            # FIXME can we make this cheaper?
            prec_aa = self._precision[..., a[..., None], a]
            prec_ba = self._precision[..., b[..., None], a]
            prec_bb = self._precision[..., b[..., None], b]
            prec_b = ops.cholesky(prec_bb)
            prec_a = ops.triangular_solve(prec_ba, prec_b)
            prec_at = ops.transpose(prec_a, -1, -2)
            precision = prec_aa - ops.matmul(prec_at, prec_a)
            prec_sqrt = ops.cholesky(precision)  # FIXME is this full rank?
            ###############################################################

            info_a = self.info_vec[..., a]
            info_b = self.info_vec[..., b]
            b_tmp = ops.triangular_solve(info_b[..., None], prec_b)
            info_vec = info_a - ops.matmul(prec_at, b_tmp)[..., 0]

            log_prob = Tensor(
                0.5 * len(b) * math.log(2 * math.pi) - _log_det_tri(prec_b),
                int_inputs,
            )
            result = log_prob + Gaussian(white_vec, prec_sqrt, inputs, False)

            return result.reduce(ops.logaddexp, reduced_ints)

        elif op is ops.add:
            # Fuse Gaussians along a plate. Compare to eager_add_gaussian_gaussian().
            inputs = OrderedDict()
            old_ints = OrderedDict()
            new_ints = OrderedDict()
            kept_perm = []
            reduced_perm = []
            for i, (k, v) in enumerate(self.inputs.items()):
                if k not in reduced_vars:
                    inputs[k] = v
                if v.dtype == "real":
                    if v in reduced_vars:
                        raise ValueError(
                            f"Cannot sum along a real dimension: {repr(v)}"
                        )
                else:
                    old_ints[k] = v
                    if k in reduced_vars:
                        reduced_perm.append(i)
                    else:
                        kept_perm.append(i)
                        new_ints[k] = v
            n = len(kept_perm) + len(reduced_perm)

            # The square root information filter fuses via transpose and reshape.
            perm = kept_perm + reduced_perm + [n]
            white_vec = ops.permute(self.white_vec, perm)
            white_vec = white_vec.reshape(white_vec.shape[: len(kept_perm)] + (-1,))
            perm = kept_perm + [n] + reduced_perm + [n + 1]
            prec_sqrt = ops.permute(self.prec_sqrt, perm)
            prec_sqrt = prec_sqrt.reshape(prec_sqrt.shape[: len(kept_perm) + 1] + (-1,))
            assert prec_sqrt.shape[:-2] == white_vec.shape[:-1]
            assert prec_sqrt.shape[-1] == white_vec.shape[-1]

            return Gaussian(white_vec, prec_sqrt, inputs, self.negate)

        return None  # defer to default implementation

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        if not self.is_full_rank:
            raise ValueError(
                "Cannot sample from a rank deficient Gaussian. "
                "Consider adding a prior."
            )

        sampled_vars = sampled_vars.intersection(self.inputs)
        if not sampled_vars:
            return self
        if any(self.inputs[k].dtype != "real" for k in sampled_vars):
            raise ValueError(
                "Sampling from non-normalized Gaussian mixtures is intentionally "
                "not implemented. You probably want to normalize. To work around, "
                "add a zero Tensor/Array with given inputs."
            )

        # Partition inputs into sample_inputs + int_inputs + real_inputs.
        sample_inputs = OrderedDict(
            (k, d) for k, d in sample_inputs.items() if k not in self.inputs
        )
        sample_shape = tuple(int(d.dtype) for d in sample_inputs.values())
        int_inputs = OrderedDict(
            (k, d) for k, d in self.inputs.items() if d.dtype != "real"
        )
        real_inputs = OrderedDict(
            (k, d) for k, d in self.inputs.items() if d.dtype == "real"
        )
        inputs = sample_inputs.copy()
        inputs.update(int_inputs)

        if sampled_vars == frozenset(real_inputs):
            shape = sample_shape + self.white_vec.shape
            backend = get_backend()
            if backend != "numpy":
                from importlib import import_module

                dist = import_module(
                    funsor.distribution.BACKEND_TO_DISTRIBUTIONS_BACKEND[backend]
                )
                sample_args = (shape,) if rng_key is None else (rng_key, shape)
                white_noise = dist.Normal.dist_class(0, 1).sample(*sample_args)
            else:
                white_noise = np.random.randn(*shape)

            sample = ops.triangular_solve(
                (white_noise + self.white_vec)[..., None],
                self._precision_chol,
                transpose=True,
            )[..., 0]
            offsets, _ = _compute_offsets(real_inputs)
            results = []
            for key, domain in real_inputs.items():
                data = sample[..., offsets[key] : offsets[key] + domain.num_elements]
                data = data.reshape(shape[:-1] + domain.shape)
                point = Tensor(data, inputs)
                assert point.output == domain
                results.append(Delta(key, point))
            results.append(self.log_normalizer)
            return reduce(ops.add, results)

        raise NotImplementedError("TODO implement partial sampling of real variables")


@eager.register(Binary, AddOp, Gaussian, Gaussian)
def eager_add_gaussian_gaussian(op, lhs, rhs):
    # Fuse two Gaussians by adding their log-densities pointwise.
    # This is similar to a Kalman filter update, but also keeps track of
    # the marginal likelihood which accumulates into a Tensor.

    # Align data.
    inputs = lhs.inputs.copy()
    inputs.update(rhs.inputs)
    lhs_white_vec, lhs_prec_sqrt = align_gaussian(inputs, lhs, expand=True)
    rhs_white_vec, rhs_prec_sqrt = align_gaussian(inputs, rhs, expand=True)

    if lhs.negate == rhs.negate:
        # Fuse aligned Gaussians.
        white_vec = ops.cat([lhs_white_vec, rhs_white_vec], -1)
        prec_sqrt = ops.cat([lhs_prec_sqrt, rhs_prec_sqrt], -1)
        return Gaussian(white_vec, prec_sqrt, inputs, lhs.negate)

    raise NotImplementedError("TODO")


@eager.register(Binary, SubOp, Gaussian, (Funsor, Align, Gaussian))
@eager.register(Binary, SubOp, (Funsor, Align, Delta), Gaussian)
def eager_sub(op, lhs, rhs):
    return lhs + -rhs


@eager.register(Unary, NegOp, Gaussian)
def eager_neg(op, arg):
    return Gaussian(arg.white_vec, arg.prec_sqrt, arg.inputs, not arg.negate)


__all__ = [
    "BlockMatrix",
    "BlockVector",
    "Gaussian",
    "align_gaussian",
]
