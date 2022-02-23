# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from functools import reduce

import funsor.ops as ops
from funsor.affine import affine_inputs, extract_affine, is_affine
from funsor.delta import Delta
from funsor.domains import Real, Reals
from funsor.interpretations import compress_gaussians
from funsor.ops import AddOp, SubOp
from funsor.tensor import Tensor, align_tensor, align_tensors
from funsor.terms import (
    Binary,
    Funsor,
    FunsorMeta,
    Number,
    Slice,
    Subs,
    Variable,
    eager,
    reflect,
)
from funsor.util import broadcast_shape, get_tracing_state, lazy_property


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


def _mmt(mat1, mat2=None):
    if mat2 is None:
        mat2 = mat1
    return mat1 @ ops.transpose(mat2, -1, -2)


def _mtm(mat1, mat2=None):
    if mat2 is None:
        mat2 = mat1
    return ops.transpose(mat1, -1, -2) @ mat2


def _inverse_cholesky(P):
    """
    Computes a Cholesky decomposition of the inverse of a posdef matrix.
    """
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = ops.cholesky(ops.flip(P, (-2, -1)))
    L_inv = ops.transpose(ops.flip(Lf, (-2, -1)), -2, -1)
    L = ops.triangular_inv(L_inv)
    return L


def _compress_rank(white_vec, prec_sqrt, assume_full_rank=False):
    """
    Compress a wide representation ``(white_vec, prec_sqrt)`` while preserving
    the quadratic function ``||x @ prec_sqrt - white_vec||^2 + const``.
    """
    dim, rank = prec_sqrt.shape[-2:]
    assert rank >= dim
    old_norm2 = _norm2(white_vec)

    # Let P = prec_sqrt and w = white_vec define the original Gaussian
    #
    #   G(x;w,P) = -1/2 || x P - w ||^2
    #            = -1/2 x P P' x' + x P w' -1/2 w w'
    #
    # We seek a compressed Gaussian G(x;wc,Pc) and constant C such that
    #
    #   G(x;w,P) = G(x;wc,Pc) + C
    #            = -1/2 x Pc Pc' x' + x Pc wc' -1/2 wc wc' + C
    if assume_full_rank:
        # Cholesky factorizing Pc = chol(P P'), we match remaining coefficients
        #
        #    Pc wc' = P w'  ==>  wc' = Pc \ P w'
        info_vec_ = prec_sqrt @ white_vec[..., None]
        precision = prec_sqrt @ ops.transpose(prec_sqrt, -1, -2)
        prec_sqrt = ops.cholesky(precision)
        white_vec = ops.triangular_solve(info_vec_, prec_sqrt)[..., 0]
    else:
        # Computing a reduced QR representation of P' of shape (rank,dim)
        #
        #   P' = Q [ R ]     P = [ R'  0 ] Q'
        #          [ 0 ]
        #
        # where Q is orthogonal and R is upper triangular of shape (dim,dim).
        # Then splitting along the new dimension,
        #
        #   G(x;w,P) = -1/2 || x [R' 0] Q' - w ||^2
        #            = -1/2 || x [R' 0] - w Q ||^2
        #            = -1/2 || x R' - (w Q)[...,:dim] ||^2
        #              -1/2 || (w Q)[...,dim:] ||^2
        #            =: G(x;wc,Pc) + C
        # where
        #
        #   wc = (w Q)[...,:dim]
        #   Pc = R'
        Q, R = ops.qr(ops.transpose(prec_sqrt, -1, -2))
        assert Q.shape[-2:] == (rank, dim)  # note only part of Q is returned
        assert R.shape[-2:] == (dim, dim)
        prec_sqrt = ops.transpose(R, -1, -2)
        white_vec = _vm(white_vec, Q)
    # Finally the shifting constant is
    #
    #    C = 1/2 (wc wc' - w w')
    new_norm2 = _norm2(white_vec)
    shift = 0.5 * (new_norm2 - old_norm2)
    return white_vec, prec_sqrt, shift


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


def _split_real_inputs(inputs, lhs_keys, prototype):
    """
    Finds a splitting set of indices ``(lhs, rhs)`` into the flat real
    dimension such that ``lhs`` indexes into real inputs in ``lhs_keys`` and
    ``rhs`` indexes into everything else.
    """
    lhs_blocks = []
    rhs_blocks = []
    start = 0
    for key, domain in inputs.items():
        if domain.dtype == "real":
            stop = start + domain.num_elements
            (lhs_blocks if key in lhs_keys else rhs_blocks).append(slice(start, stop))
            start = stop

    # There are three cases: lhs left of rhs (cheap slices), lhs right of rhs
    # (cheap slices), and interleaved (expensive advanced indexing tensors).
    lhs_start = min(b.start for b in lhs_blocks)
    rhs_start = min(b.start for b in rhs_blocks)
    lhs_stop = max(b.stop for b in lhs_blocks)
    rhs_stop = max(b.stop for b in rhs_blocks)
    if lhs_stop <= rhs_start or rhs_stop <= lhs_start:
        # Construct cheap slices.
        lhs = slice(lhs_start, lhs_stop)
        rhs = slice(rhs_start, rhs_stop)
        return lhs, rhs

    # Construct interleaving indices.
    lhs = ops.cat([ops.new_arange(prototype, b.start, b.stop) for b in lhs_blocks])
    rhs = ops.cat([ops.new_arange(prototype, b.start, b.stop) for b in rhs_blocks])
    return lhs, rhs


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
        # TODO optimize this to use backend-specific block setters:
        # .__setitem__ for numpy and torch; .at(...).set(...) for jax.

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
        # TODO optimize this to use backend-specific block setters:
        # .__setitem__ for numpy and torch; .at(...).set(...) for jax.

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
        *,
        mean=None,
        info_vec=None,
        precision=None,
        scale_tril=None,
        covariance=None,
    ):
        # Convert inputs.
        assert inputs is not None
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        assert isinstance(inputs, tuple)

        # Convert scale parameter to prec_sqrt.
        if prec_sqrt is None and white_vec is not None:
            raise ValueError("Cannot specify white_vec without prec_sqrt")
        if prec_sqrt is not None:
            is_tril = False
        elif precision is not None:
            prec_sqrt = ops.cholesky(precision)
            is_tril = True
        elif covariance is not None:
            prec_sqrt = _inverse_cholesky(covariance)
            is_tril = True
        elif scale_tril is not None:
            prec_sqrt = ops.transpose(ops.triangular_inv(scale_tril), -1, -2)
            is_tril = False
        else:
            raise ValueError(
                "At least one of prec_sqrt, precision, scale_tril, or covariance "
                "must be specified"
            )

        # Convert location parameter to white_vec.
        if white_vec is not None:
            pass
        elif mean is not None:
            white_vec = _vm(mean, prec_sqrt)
        elif info_vec is not None:
            if not is_tril:
                prec_sqrt = ops.cholesky(_mmt(prec_sqrt))  # triangularize
                is_tril = True
            white_vec = ops.triangular_solve(info_vec[..., None], prec_sqrt)[..., 0]
        else:
            raise ValueError(
                "At least one of white_vec, mean, or info_vec must be specified"
            )

        # Compress wide representations.
        shift = None
        dim, rank = prec_sqrt.shape[-2:]
        if rank > dim * cls.compression_threshold:
            white_vec, prec_sqrt, shift = _compress_rank(white_vec, prec_sqrt)

        # Create a Gaussian.
        result = super().__call__(white_vec, prec_sqrt, inputs)

        # Add compression byproducts.
        if shift is not None:
            int_inputs = OrderedDict((k, v) for k, v in inputs if v.dtype != "real")
            result += Tensor(shift, int_inputs)

        return result


class Gaussian(Funsor, metaclass=GaussianMeta):
    r"""
    Funsor representing a batched Gaussian log-density function.

    Gaussians are the internal representation for joint and conditional
    multivariate normal distributions and multivariate normal likelihoods.
    Mathematically, a Gaussian represents the quadratic log density function::

        f(x) = -0.5 * || x @ prec_sqrt - white_vec ||^2
             = -0.5 * < x @ prec_sqrt - white_vec | x @ prec_sqrt - white_vec >
             = -0.5 * < x | prec_sqrt @ prec_sqrt.T | x>
               + < x | prec_sqrt | white_vec > - 0.5 ||white_vec||^2

    Internally Gaussians use a square root information filter (SRIF)
    representation consisting of a square root of the precision matrix
    ``prec_sqrt`` and a vector in the whitened space ``white_vec``. This
    representation allows space-efficient construction of Gaussians with
    incomplete information, i.e. with zero eigenvalues in the precision matrix.
    These incomplete log densities arise when making low-dimensional
    observations of higher-dimensional hidden state. Sampling and
    marginalization are supported only for full-rank Gaussians or full-rank
    subsets of Gaussians. See the :meth:`rank` and :meth:`is_full_rank`
    properties.

    .. note:: :class:`Gaussian` s are not normalized probability distributions,
        rather they are canonicalized to evaluate to zero log density at their
        maximum: ``f(prec_sqrt \ white_vec) = 0``. Not only are Gaussians
        non-normalized, but they may be rank deficient and non-normalizable, in
        which case sampling and marginalization are supported only un full-rank
        subsets of variables.

    :param torch.Tensor white_vec: An batched white noise vector, where
        ``white_vec = prec_sqrt.T @ mean``. Alternatively you can specify one
        of the kwargs ``mean`` or ``info_vec``, which will be converted to
        ``white_vec``.
    :param torch.Tensor prec_sqrt: A batched square root of the positive
        semidefinite precision matrix. This need not be square, and typically
        has shape ``prec_sqrt.shape == white_vec.shape[:-1] + (dim, rank)``,
        where ``dim`` is the total flattened size of real inputs and
        ``rank = white_vec.shape[-1]``.  Alternatively you can specify one of
        the kwargs ``precision``, ``covariance``, or ``scale_tril``, which will
        be converted to ``prec_sqrt``.
    :param OrderedDict inputs: Mapping from name to
        :class:`~funsor.domains.Domain` .
    """

    compression_threshold = 2

    def __init__(self, white_vec, prec_sqrt, inputs):
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
            # This should be true but weirdly fails in pytest tests that use
            # set_compression_threshold(large_value).
            # assert rank <= dim * self.compression_threshold

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
        self.batch_shape = batch_shape
        self.event_shape = (dim,)

    @classmethod
    @contextmanager
    def set_compression_threshold(cls, threshold: float):
        """
        Context manager to set rank compression threshold.

        To save space Gaussians compress wide ``prec_sqrt`` matrices down to
        square. However compression uses a QR decomposition which can be
        expensive and which has unstable gradients when the resulting precision
        matrix is rank deficient. To balance space and time costs and numerical
        stability, compression is trigger only on ``prec_sqrt`` matrices whose
        width to height ratio is greater than ``threshold``.

        :param float threshold: Defaults to 2. To optimize for space and
            deterministic computations, set ``threshold = 1``. To optimize for
            fewest QR decompositions and numerical stability, set ``threshold =
            math.inf``.
        """
        assert isinstance(threshold, (int, float))
        assert threshold >= 1
        old = cls.compression_threshold
        try:
            cls.compression_threshold = threshold
            yield
        finally:
            cls.compression_threshold = old

    def __repr__(self):
        return "Gaussian(..., ({}))".format(
            " ".join("({}, {}),".format(*kv) for kv in self.inputs.items())
        )

    @property
    def rank(self):
        return self.prec_sqrt.shape[-1]

    @property
    def is_full_rank(self):
        dim, rank = self.prec_sqrt.shape[-2:]
        return rank >= dim

    # TODO Consider weak-memoizing these so they persist through alpha conversion.
    # https://github.com/pyro-ppl/pyro/blob/ac3c588/pyro/distributions/coalescent.py#L412
    @lazy_property
    def _precision(self):
        return self.prec_sqrt @ ops.transpose(self.prec_sqrt, -1, -2)

    @lazy_property
    def _precision_chol(self):
        # Note self.prec_sqrt may be neither lower triangular nor square.
        assert self.is_full_rank
        return ops.cholesky(self._precision)

    @lazy_property
    def _covariance(self):
        return ops.cholesky_inverse(self._precision_chol)

    @lazy_property
    def _scale_tril(self):
        return _inverse_cholesky(self._precision)

    @lazy_property
    def _mean(self):
        return ops.cholesky_solve(self._info_vec[..., None], self._precision_chol)[
            ..., 0
        ]

    @lazy_property
    def _info_vec(self):
        return _mv(self.prec_sqrt, self.white_vec)

    @lazy_property
    def _log_normalizer(self):
        dim = self.prec_sqrt.shape[-2]
        log_det_term = _log_det_tri(self._precision_chol)
        result = 0.5 * dim * math.log(2 * math.pi) - log_det_term
        if self.rank == dim:
            return result
        # Shift, as in logic in _compress_rank().
        old_norm2 = _norm2(self.white_vec)
        white_vec = ops.triangular_solve(
            self._info_vec[..., None], self._precision_chol
        )[..., 0]
        new_norm2 = _norm2(white_vec)
        shift = 0.5 * (new_norm2 - old_norm2)
        return result + shift

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
        return Gaussian(white_vec, prec_sqrt, inputs)

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
        var_result = Gaussian(self.white_vec, self.prec_sqrt, inputs)
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
        int_result = Gaussian(funsors[0].data, funsors[1].data, inputs)
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
            result = -0.5 * _norm2(_vm(value, prec_sqrt) - white_vec)
            result = Tensor(result, int_inputs)
            assert result.output == Real
            return Subs(result, remaining_subs) if remaining_subs else result

        # Perform a partial substution of a subset of real variables, resulting
        # in a Joint. We split real inputs into two sets: a for the preserved
        # and b for the substituted.
        #   G([xa xb]; w, [ Pa ]) = -0.5 || xa Pa + xb Pb - w||2
        #                 [ Pb ]
        #                         = G(xa; w - xb Pb, Pa)
        # where  wa := w - xb Pb  is the new white_vec.
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
        result = Gaussian(white_vec_a, prec_sqrt_a, inputs)
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

        # Construct the new Gaussian. Suppose the old Gaussian funsor g has density
        #   G(x;w,P) = -1/2 || x P - w ||^2
        # Now define a new Gaussian by substituting x = A y + b:
        #   G(y;w',P') = G(y A + b; w, P)
        #              = -1/2 || (y A + b) P - w ||^2
        #              = -1/2 || y A P - w + b P ||^2
        #              =: -1/2 || y P' - w' ||^2
        #              = G(y; w - b P, A P)
        # where  P' = A P  and  w' = w - b P  parametrize the new Gaussian.
        white_vec = white_vec - _vm(subs_vector, prec_sqrt)
        prec_sqrt = subs_matrix @ prec_sqrt
        result = Gaussian(white_vec, prec_sqrt, new_inputs)
        return Subs(result, remaining_subs) if remaining_subs else result

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if op is ops.logaddexp:
            # Marginalize out real variables, but keep mixtures lazy.
            assert all(v in self.inputs for v in reduced_vars)
            real_vars = frozenset(
                k for k, d in self.inputs.items() if d.dtype == "real"
            )
            reduced_reals = reduced_vars & real_vars
            reduced_ints = reduced_vars - real_vars
            if not reduced_reals:
                return None  # defer to default implementation
            if reduced_reals == real_vars:
                return self.log_normalizer.reduce(ops.logaddexp, reduced_ints)

            inputs = OrderedDict(
                (k, d) for k, d in self.inputs.items() if k not in reduced_reals
            )
            int_inputs = OrderedDict(
                (k, v) for k, v in inputs.items() if v.dtype != "real"
            )

            # Let x = [xa xb] where xb will be marginalized out and will xa
            # remain. Following the formula in _compress_rank, we can rewrite
            # the joint Gaussian as a Gaussian in xb plus a term C that does
            # not depend on xb:
            #
            #   log integral exp G([xa xb]; w, [ Pa ]) dxb
            #                                  [ Pb ]
            #     =: G(xb; wb, Qb).log_normalizer + C
            #
            # In normalizable models, rank >= dim(xb), so we can choose Qb to
            # be the Cholesky square root, making it easy to compute a
            # determinant and solve for wb.
            #
            #    Qb = chol(Pb Pb')
            #   wb' = Qb \ Pb (w - xa Pa)'
            #
            # Next we match moments of C to a Gaussian in xa:
            #
            #   C = 1/2 (wb wb' - w w')  # from _compress_rank
            #     = 1/2 (xa Pa - w) Pb' inv(Qb Qb') Pb (xa Pa - w)'
            #     - 1/2 (xa Pa - w) (xa Pa - w)'
            #     =: G(xa; wa, Qa)
            #
            # whence  Qa = Pa S  and  wa = w S, where S is a square root of the
            # rank-by-rank projection matrix (S = S S' by idempotence):
            #
            #   S S' = I - Pb' inv(Pb Pb') Pb = I - (Qb\Pb)' (Qb\Pb) = S
            #
            # Note if rank == dim(xb), then the projection matrix is zero,
            # and the Gaussian G(xa; wa, Qa) is zero can be dropped.
            b, a = _split_real_inputs(self.inputs, reduced_vars, self.white_vec)
            prec_sqrt_a = self.prec_sqrt[..., a, :]
            prec_sqrt_b = self.prec_sqrt[..., b, :]
            dim_b = prec_sqrt_b.shape[-2]
            if self.rank < dim_b:
                raise ValueError(
                    f"Too little information to marginalize over {set(reduced_vars)}. "
                    "Consider adding a prior."
                )
            precision_chol_b = ops.cholesky(_mmt(prec_sqrt_b))  # assume full rank
            result = self._marginalize_after_split(
                inputs, int_inputs, prec_sqrt_b, prec_sqrt_a, precision_chol_b
            )
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

            return Gaussian(white_vec, prec_sqrt, inputs)

        return None  # defer to default implementation

    def _sample(self, sampled_vars, sample_inputs, rng_key):
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
        int_inputs = OrderedDict()
        sampled_real_inputs = OrderedDict()
        remaining_real_inputs = OrderedDict()
        for k, d in self.inputs.items():
            if d.dtype != "real":
                int_inputs[k] = d
            elif k in sampled_vars:
                sampled_real_inputs[k] = d
            else:
                remaining_real_inputs[k] = d
        if self.rank < sum(d.num_elements for d in sampled_real_inputs.values()):
            raise ValueError(
                f"Too little information to sample over {set(sampled_vars)}. "
                "Consider adding a prior."
            )

        if not remaining_real_inputs:  # Sample all variables.
            # Triangularize via _compress_rank().
            white_vec, prec_sqrt, _ = _compress_rank(
                self.white_vec, self.prec_sqrt, assume_full_rank=True
            )

            # Jointly sample.
            # This section may involve either Funsors or backend arrays.
            dim = prec_sqrt.shape[-1]
            white_noise = _sample_white_noise(
                sample_inputs, int_inputs, dim, self.white_vec, rng_key
            )
            if isinstance(white_noise, Funsor):
                white_vec = Tensor(white_vec, int_inputs)
                prec_sqrt = Tensor(prec_sqrt, int_inputs)
            sample = ops.triangular_solve(
                (white_noise + white_vec)[..., None], prec_sqrt, transpose=True
            )[..., 0]

            # Compute the remaining Tensor.
            remaining = self.log_normalizer

        else:  # Sample only a subset of real variables.
            # Split into sampled variables a and remaining variables b.
            a, b = _split_real_inputs(self.inputs, sampled_vars, self.white_vec)
            prec_sqrt_a = self.prec_sqrt[..., a, :]
            prec_sqrt_b = self.prec_sqrt[..., b, :]
            dim_a = prec_sqrt_a.shape[-2]

            # Compute white_vec of a lazily conditioned on b's variables.
            # This requires Funsors rather than backend arrays.
            flat = ops.cat(
                [
                    Variable(k, d).reshape((d.num_elements,))
                    for k, d in remaining_real_inputs.items()
                ]
            )
            white_vec_a = (
                Tensor(self.white_vec, int_inputs)
                - (flat[None] @ Tensor(prec_sqrt_b, int_inputs))[0]
            )

            # Triangularize.
            precision_chol_a = Tensor(ops.cholesky(_mmt(prec_sqrt_a)), int_inputs)
            white_vec_a = ops.triangular_solve(
                Tensor(prec_sqrt_a, int_inputs) @ white_vec_a[..., None],
                precision_chol_a,
            )[..., 0]

            # Jointly sample.
            white_noise = _sample_white_noise(
                sample_inputs, int_inputs, dim_a, self.white_vec, rng_key
            )
            if not isinstance(white_noise, Funsor):
                inputs = sample_inputs.copy()
                inputs.update(int_inputs)
                white_noise = Tensor(white_noise, inputs)
            sample = ops.triangular_solve(
                (white_noise + white_vec_a)[..., None], precision_chol_a, transpose=True
            )[..., 0]

            # Compute the remaining Gaussian, equivalent to
            # self.reduce(ops.logaddexp, sampled_vars), but avoiding duplicate work.
            inputs = int_inputs.copy()
            inputs.update(remaining_real_inputs)
            remaining = self._marginalize_after_split(
                inputs, int_inputs, prec_sqrt_a, prec_sqrt_b, precision_chol_a.data
            )

        # Extract shaped components of the flat concatenated sample.
        results = [remaining]
        offsets, _ = _compute_offsets(sampled_real_inputs)
        for key, domain in sampled_real_inputs.items():
            point = sample[..., offsets[key] : offsets[key] + domain.num_elements]
            point = point.reshape(point.shape[:-1] + domain.shape)
            if not isinstance(point, Funsor):  # I.e. when eagerly sampling.
                inputs = sample_inputs.copy()
                inputs.update(int_inputs)
                point = Tensor(point, inputs)
            assert point.output == domain
            results.append(Delta(key, point))

        return reduce(ops.add, results)

    def _marginalize_after_split(
        self, inputs, int_inputs, prec_sqrt_a, prec_sqrt_b, precision_chol_a
    ):
        """
        Helper used in partial reduction and partial sampling.
        This marginalizes over a and returns a shifted Gaussian over b.
        """
        dim_a = prec_sqrt_a.shape[-2]
        dim_b = prec_sqrt_b.shape[-2]
        result = Tensor(
            dim_a * math.log(2 * math.pi) / 2 - _log_det_tri(precision_chol_a),
            int_inputs,
        )
        if self.rank > dim_a:
            proj_a = _mtm(ops.triangular_solve(prec_sqrt_a, precision_chol_a))
            prec_sqrt = prec_sqrt_b - prec_sqrt_b @ proj_a
            white_vec = self.white_vec - _vm(self.white_vec, proj_a)
            result += Gaussian(white_vec, prec_sqrt, inputs)
        else:  # The Gaussian over xa is zero.
            # TODO switch from an empty Gaussian to a Constant once this works:
            # from .constant import Constant
            # const_inputs = OrderedDict(
            #     (k, v) for k, v in inputs.items() if k not in result.inputs
            # )
            # result = Constant(const_inputs, result)
            batch_shape = self.white_vec.shape[:-1]
            white_vec = ops.new_zeros(self.white_vec, batch_shape + (0,))
            prec_sqrt = ops.new_zeros(self.white_vec, batch_shape + (dim_b, 0))
            result += Gaussian(white_vec, prec_sqrt, inputs)
        return result


def _sample_white_noise(sample_inputs, int_inputs, dim, prototype, rng_key):
    if [v.dtype for v in sample_inputs.values()] == ["real"]:
        # Lazily compute a sample as a function of white noise.
        k, d = next(iter(sample_inputs.items()))
        return Variable(k, d)[tuple(int_inputs)]

    # Eagerly draw noise.
    shape = tuple(d.size for d in sample_inputs.values() if d.dtype != "real")
    shape += tuple(d.size for d in int_inputs.values())
    shape += (dim,)
    assert ops.is_numeric_array(prototype)
    return ops.randn(prototype, shape, rng_key)


@compress_gaussians.register(Gaussian, object, object, tuple)
def _compress_gaussians(white_vec, prec_sqrt, inputs):
    dim, rank = prec_sqrt.shape[-2:]
    if rank <= dim:
        return None
    white_vec, prec_sqrt, shift = _compress_rank(white_vec, prec_sqrt)
    int_inputs = OrderedDict((k, v) for k, v in inputs if v.dtype != "real")
    return Gaussian(white_vec, prec_sqrt, inputs) + Tensor(shift, int_inputs)


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

    # Fuse aligned Gaussians via concatenation.
    white_vec = ops.cat([lhs_white_vec, rhs_white_vec], -1)
    prec_sqrt = ops.cat([lhs_prec_sqrt, rhs_prec_sqrt], -1)
    return Gaussian(white_vec, prec_sqrt, inputs)


@eager.register(Binary, SubOp, Gaussian, Gaussian)
def eager_sub(op, lhs, rhs):
    return lhs + (-rhs)


__all__ = [
    "BlockMatrix",
    "BlockVector",
    "Gaussian",
    "align_gaussian",
]
