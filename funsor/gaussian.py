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
from funsor.terms import Align, Binary, Funsor, FunsorMeta, Number, Slice, Subs, Unary, Variable, eager, reflect
from funsor.util import broadcast_shape, get_backend, get_tracing_state, lazy_property


def _log_det_tri(x):
    return ops.log(ops.diagonal(x, -1, -2)).sum(-1)


def _vv(vec1, vec2):
    """
    Computes the inner product ``< vec1 | vec 2 >``.
    """
    return ops.matmul(ops.unsqueeze(vec1, -2), ops.unsqueeze(vec2, -1)).squeeze(-1).squeeze(-1)


def _mv(mat, vec):
    return ops.matmul(mat, ops.unsqueeze(vec, -1)).squeeze(-1)


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
        if domain.dtype == 'real':
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
                self.parts[i] = ops.new_zeros(prototype, self.shape[:-1] + (i[1] - i[0],))

        # Concatenate parts.
        parts = [v for k, v in sorted(self.parts.items())]
        result = ops.cat(-1, *parts)
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
        columns = {i: ops.cat(-1, *[v for j, v in sorted(part.items())])
                   for i, part in self.parts.items()}
        result = ops.cat(-2, *[v for i, v in sorted(columns.items())])
        if not get_tracing_state():
            assert result.shape == self.shape
        return result


def align_gaussian(new_inputs, old):
    """
    Align data of a Gaussian distribution to a new ``inputs`` shape.
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(old, Gaussian)
    info_vec = old.info_vec
    precision = old.precision

    # Align int inputs.
    # Since these are are managed as in Tensor, we can defer to align_tensor().
    new_ints = OrderedDict((k, d) for k, d in new_inputs.items() if d.dtype != 'real')
    old_ints = OrderedDict((k, d) for k, d in old.inputs.items() if d.dtype != 'real')
    if new_ints != old_ints:
        info_vec = align_tensor(new_ints, Tensor(info_vec, old_ints))
        precision = align_tensor(new_ints, Tensor(precision, old_ints))

    # Align real inputs, which are all concatenated in the rightmost dims.
    new_offsets, new_dim = _compute_offsets(new_inputs)
    old_offsets, old_dim = _compute_offsets(old.inputs)
    assert info_vec.shape[-1:] == (old_dim,)
    assert precision.shape[-2:] == (old_dim, old_dim)
    if new_offsets != old_offsets:
        old_info_vec = info_vec
        old_precision = precision
        info_vec = BlockVector(old_info_vec.shape[:-1] + (new_dim,))
        precision = BlockMatrix(old_info_vec.shape[:-1] + (new_dim, new_dim))
        for k1, new_offset1 in new_offsets.items():
            if k1 not in old_offsets:
                continue
            offset1 = old_offsets[k1]
            num_elements1 = old.inputs[k1].num_elements
            old_slice1 = slice(offset1, offset1 + num_elements1)
            new_slice1 = slice(new_offset1, new_offset1 + num_elements1)
            info_vec[..., new_slice1] = old_info_vec[..., old_slice1]
            for k2, new_offset2 in new_offsets.items():
                if k2 not in old_offsets:
                    continue
                offset2 = old_offsets[k2]
                num_elements2 = old.inputs[k2].num_elements
                old_slice2 = slice(offset2, offset2 + num_elements2)
                new_slice2 = slice(new_offset2, new_offset2 + num_elements2)
                precision[..., new_slice1, new_slice2] = old_precision[..., old_slice1, old_slice2]
        info_vec = info_vec.as_tensor()
        precision = precision.as_tensor()

    return info_vec, precision


class GaussianMeta(FunsorMeta):
    """
    Wrapper to convert between OrderedDict and tuple.
    """
    def __call__(cls, info_vec, precision, inputs):
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        assert isinstance(inputs, tuple)
        return super(GaussianMeta, cls).__call__(info_vec, precision, inputs)


class Gaussian(Funsor, metaclass=GaussianMeta):
    """
    Funsor representing a batched joint Gaussian distribution as a log-density
    function.

    Mathematically, a Gaussian represents the density function::

        f(x) = < x | info_vec > - 0.5 * < x | precision | x >
             = < x | info_vec - 0.5 * precision @ x >

    Note that :class:`Gaussian` s are not normalized, rather they are
    canonicalized to evaluate to zero log density at the origin: ``f(0) = 0``.
    This canonical form is useful in combination with the information filter
    representation because it allows :class:`Gaussian` s with incomplete
    information, i.e.  zero eigenvalues in the precision matrix.  These
    incomplete distributions arise when making low-dimensional observations on
    higher dimensional hidden state.

    :param torch.Tensor info_vec: An optional batched information vector,
        where ``info_vec = precision @ mean``.
    :param torch.Tensor precision: A batched positive semidefinite precision
        matrix.
    :param OrderedDict inputs: Mapping from name to
        :class:`~funsor.domains.Domain` .
    """
    def __init__(self, info_vec, precision, inputs):
        assert ops.is_numeric_array(info_vec) and ops.is_numeric_array(precision)
        assert isinstance(inputs, tuple)
        inputs = OrderedDict(inputs)

        # Compute total dimension of all real inputs.
        dim = sum(d.num_elements for d in inputs.values() if d.dtype == 'real')
        if not get_tracing_state():
            assert dim
            assert len(precision.shape) >= 2 and precision.shape[-2:] == (dim, dim)
            assert len(info_vec.shape) >= 1 and info_vec.shape[-1] == dim

        # Compute total shape of all Bint inputs.
        batch_shape = tuple(d.dtype for d in inputs.values()
                            if isinstance(d.dtype, int))
        if not get_tracing_state():
            assert precision.shape == batch_shape + (dim, dim)
            assert info_vec.shape == batch_shape + (dim,)

        output = Real
        fresh = frozenset(inputs.keys())
        bound = {}
        super(Gaussian, self).__init__(inputs, output, fresh, bound)
        self.info_vec = info_vec
        self.precision = precision
        self.batch_shape = batch_shape
        self.event_shape = (dim,)

    @lazy_property
    def _precision_chol(self):
        return ops.cholesky(self.precision)

    @lazy_property
    def log_normalizer(self):
        dim = self.precision.shape[-1]
        log_det_term = _log_det_tri(self._precision_chol)
        loc_info_vec_term = 0.5 * (ops.triangular_solve(
            self.info_vec[..., None], self._precision_chol)[..., 0] ** 2).sum(-1)
        data = 0.5 * dim * math.log(2 * math.pi) - log_det_term + loc_info_vec_term
        inputs = OrderedDict((k, v) for k, v in self.inputs.items() if v.dtype != 'real')
        return Tensor(data, inputs)

    def __repr__(self):
        return 'Gaussian(..., ({}))'.format(' '.join(
            '({}, {}),'.format(*kv) for kv in self.inputs.items()))

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self

        inputs = OrderedDict((name, self.inputs[name]) for name in names)
        inputs.update(self.inputs)
        info_vec, precision = align_gaussian(inputs, self)
        return Gaussian(info_vec, precision, inputs)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        prototype = Tensor(self.info_vec)
        subs = tuple((k, v if isinstance(v, (Variable, Slice))
                      else prototype.materialize(v))
                     for k, v in subs if k in self.inputs)
        if not subs:
            return self

        # Constants and Affine funsors are eagerly substituted;
        # everything else is lazily substituted.
        lazy_subs = tuple((k, v) for k, v in subs
                          if not isinstance(v, (Number, Tensor, Variable, Slice))
                          and not (is_affine(v) and affine_inputs(v)))
        var_subs = tuple((k, v) for k, v in subs if isinstance(v, Variable))
        int_subs = tuple((k, v) for k, v in subs if isinstance(v, (Number, Tensor, Slice))
                         if v.dtype != 'real')
        real_subs = tuple((k, v) for k, v in subs if isinstance(v, (Number, Tensor))
                          if v.dtype == 'real')
        affine_subs = tuple((k, v) for k, v in subs
                            if is_affine(v) and affine_inputs(v) and not isinstance(v, Variable))
        if var_subs:
            return self._eager_subs_var(var_subs, int_subs + real_subs + affine_subs + lazy_subs)
        if int_subs:
            return self._eager_subs_int(int_subs, real_subs + affine_subs + lazy_subs)
        if real_subs:
            return self._eager_subs_real(real_subs, affine_subs + lazy_subs)
        if affine_subs:
            return self._eager_subs_affine(affine_subs, lazy_subs)
        return reflect(Subs, self, lazy_subs)

    def _eager_subs_var(self, subs, remaining_subs):
        # Perform variable substitution, i.e. renaming of inputs.
        rename = {k: v.name for k, v in subs}
        inputs = OrderedDict((rename.get(k, k), d) for k, d in self.inputs.items())
        if len(inputs) != len(self.inputs):
            raise ValueError("Variable substitution name conflict")
        var_result = Gaussian(self.info_vec, self.precision, inputs)
        return Subs(var_result, remaining_subs) if remaining_subs else var_result

    def _eager_subs_int(self, subs, remaining_subs):
        # Perform integer substitution, i.e. slicing into a batch.
        int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
        real_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype == 'real')
        tensors = [self.info_vec, self.precision]
        funsors = [Subs(Tensor(x, int_inputs), subs) for x in tensors]
        inputs = funsors[0].inputs.copy()
        inputs.update(real_inputs)
        int_result = Gaussian(funsors[0].data, funsors[1].data, inputs)
        return Subs(int_result, remaining_subs) if remaining_subs else int_result

    def _eager_subs_real(self, subs, remaining_subs):
        # Broadcast all component tensors.
        subs = OrderedDict(subs)
        int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
        tensors = [Tensor(self.info_vec, int_inputs),
                   Tensor(self.precision, int_inputs)]
        tensors.extend(subs.values())
        int_inputs, tensors = align_tensors(*tensors)
        batch_dim = len(tensors[0].shape) - 1
        batch_shape = broadcast_shape(*(x.shape[:batch_dim] for x in tensors))
        (info_vec, precision), values = tensors[:2], tensors[2:]
        offsets, event_size = _compute_offsets(self.inputs)
        slices = [(k, slice(offset, offset + self.inputs[k].num_elements))
                  for k, offset in offsets.items()]

        # Expand all substituted values.
        values = OrderedDict(zip(subs, values))
        for k, value in values.items():
            value = value.reshape(value.shape[:batch_dim] + (-1,))
            if not get_tracing_state():
                assert value.shape[-1] == self.inputs[k].num_elements
            values[k] = ops.expand(value, batch_shape + value.shape[-1:])

        # Try to perform a complete substitution of all real variables, resulting in a Tensor.
        if all(k in subs for k, d in self.inputs.items() if d.dtype == 'real'):
            # Form the concatenated value.
            value = BlockVector(batch_shape + (event_size,))
            for k, i in slices:
                if k in values:
                    value[..., i] = values[k]
            value = value.as_tensor()

            # Evaluate the non-normalized log density.
            result = _vv(value, info_vec - 0.5 * _mv(precision, value))

            result = Tensor(result, int_inputs)
            assert result.output == Real
            return Subs(result, remaining_subs) if remaining_subs else result

        # Perform a partial substution of a subset of real variables, resulting in a Joint.
        # We split real inputs into two sets: a for the preserved and b for the substituted.
        b = frozenset(k for k, v in subs.items())
        a = frozenset(k for k, d in self.inputs.items() if d.dtype == 'real' and k not in b)
        prec_aa = ops.cat(-2, *[ops.cat(-1, *[
            precision[..., i1, i2]
            for k2, i2 in slices if k2 in a])
            for k1, i1 in slices if k1 in a])
        prec_ab = ops.cat(-2, *[ops.cat(-1, *[
            precision[..., i1, i2]
            for k2, i2 in slices if k2 in b])
            for k1, i1 in slices if k1 in a])
        prec_bb = ops.cat(-2, *[ops.cat(-1, *[
            precision[..., i1, i2]
            for k2, i2 in slices if k2 in b])
            for k1, i1 in slices if k1 in b])
        info_a = ops.cat(-1, *[info_vec[..., i] for k, i in slices if k in a])
        info_b = ops.cat(-1, *[info_vec[..., i] for k, i in slices if k in b])
        value_b = ops.cat(-1, *[values[k] for k, i in slices if k in b])
        info_vec = info_a - _mv(prec_ab, value_b)
        log_scale = _vv(value_b, info_b - 0.5 * _mv(prec_bb, value_b))
        precision = ops.expand(prec_aa, info_vec.shape + info_vec.shape[-1:])
        inputs = int_inputs.copy()
        for k, d in self.inputs.items():
            if k not in subs:
                inputs[k] = d
        result = Gaussian(info_vec, precision, inputs) + Tensor(log_scale, int_inputs)
        return Subs(result, remaining_subs) if remaining_subs else result

    def _eager_subs_affine(self, subs, remaining_subs):
        # Extract an affine representation.
        affine = OrderedDict()
        for k, v in subs:
            const, coeffs = extract_affine(v)
            if (isinstance(const, Tensor) and
                    all(isinstance(coeff, Tensor) for coeff, _ in coeffs.values())):
                affine[k] = const, coeffs
            else:
                remaining_subs += (k, v),
        if not affine:
            return reflect(Subs, self, remaining_subs)

        # Align integer dimensions.
        old_int_inputs = OrderedDict((k, v) for k, v in self.inputs.items() if v.dtype != 'real')
        tensors = [Tensor(self.info_vec, old_int_inputs),
                   Tensor(self.precision, old_int_inputs)]
        for const, coeffs in affine.values():
            tensors.append(const)
            tensors.extend(coeff for coeff, _ in coeffs.values())
        new_int_inputs, tensors = align_tensors(*tensors, expand=True)
        tensors = (Tensor(x, new_int_inputs) for x in tensors)
        old_info_vec = next(tensors).data
        old_precision = next(tensors).data
        for old_k, (const, coeffs) in affine.items():
            const = next(tensors)
            for new_k, (coeff, eqn) in coeffs.items():
                coeff = next(tensors)
                coeffs[new_k] = coeff, eqn
            affine[old_k] = const, coeffs
        batch_shape = old_info_vec.shape[:-1]

        # Align real dimensions.
        old_real_inputs = OrderedDict((k, v) for k, v in self.inputs.items() if v.dtype == 'real')
        new_real_inputs = old_real_inputs.copy()
        for old_k, (const, coeffs) in affine.items():
            del new_real_inputs[old_k]
            for new_k, (coeff, eqn) in coeffs.items():
                new_shape = coeff.shape[:len(eqn.split('->')[0].split(',')[1])]
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
                subs_matrix[..., new_slice, old_slice] = \
                    ops.new_eye(self.info_vec, batch_shape + (old_size,))
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
                    subs_matrix[..., new_slice, old_slice] = \
                        coeff.data.reshape(batch_shape + (new_size, old_size))
        subs_vector = subs_vector.as_tensor()
        subs_matrix = subs_matrix.as_tensor()
        subs_matrix_t = ops.transpose(subs_matrix, -1, -2)

        # Construct the new funsor. Suppose the old Gaussian funsor g has density
        #   g(x) = < x | i - 1/2 P x>
        # Now define a new funsor f by substituting x = A y + B:
        #   f(y) = g(A y + B)
        #        = < A y + B | i - 1/2 P (A y + B) >
        #        = < y | At (i - P B) - 1/2 At P A y > + < B | i - 1/2 P B >
        #        =: < y | i' - 1/2 P' y > + C
        # where  P' = At P A  and  i' = At (i - P B)  parametrize a new Gaussian
        # and  C = < B | i - 1/2 P B >  parametrize a new Tensor.
        precision = subs_matrix @ old_precision @ subs_matrix_t
        info_vec = _mv(subs_matrix, old_info_vec - _mv(old_precision, subs_vector))
        const = _vv(subs_vector, old_info_vec - 0.5 * _mv(old_precision, subs_vector))
        result = Gaussian(info_vec, precision, new_inputs) + Tensor(const, new_int_inputs)
        return Subs(result, remaining_subs) if remaining_subs else result

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if op is ops.logaddexp:
            # Marginalize out real variables, but keep mixtures lazy.
            assert all(v in self.inputs for v in reduced_vars)
            real_vars = frozenset(k for k, d in self.inputs.items() if d.dtype == "real")
            reduced_reals = reduced_vars & real_vars
            reduced_ints = reduced_vars - real_vars
            if not reduced_reals:
                return None  # defer to default implementation

            inputs = OrderedDict((k, d) for k, d in self.inputs.items() if k not in reduced_reals)
            if reduced_reals == real_vars:
                result = self.log_normalizer
            else:
                int_inputs = OrderedDict((k, v) for k, v in inputs.items() if v.dtype != 'real')
                offsets, _ = _compute_offsets(self.inputs)
                a = []
                b = []
                for key, domain in self.inputs.items():
                    if domain.dtype == 'real':
                        block = ops.new_arange(self.info_vec, offsets[key], offsets[key] + domain.num_elements, 1)
                        (b if key in reduced_vars else a).append(block)
                a = ops.cat(-1, *a)
                b = ops.cat(-1, *b)
                prec_aa = self.precision[..., a[..., None], a]
                prec_ba = self.precision[..., b[..., None], a]
                prec_bb = self.precision[..., b[..., None], b]
                prec_b = ops.cholesky(prec_bb)
                prec_a = ops.triangular_solve(prec_ba, prec_b)
                prec_at = ops.transpose(prec_a, -1, -2)
                precision = prec_aa - ops.matmul(prec_at, prec_a)

                info_a = self.info_vec[..., a]
                info_b = self.info_vec[..., b]
                b_tmp = ops.triangular_solve(info_b[..., None], prec_b)
                info_vec = info_a - ops.matmul(prec_at, b_tmp)[..., 0]

                log_prob = Tensor(0.5 * len(b) * math.log(2 * math.pi) - _log_det_tri(prec_b) +
                                  0.5 * (b_tmp[..., 0] ** 2).sum(-1),
                                  int_inputs)
                result = log_prob + Gaussian(info_vec, precision, inputs)

            return result.reduce(ops.logaddexp, reduced_ints)

        elif op is ops.add:
            for v in reduced_vars:
                if self.inputs[v].dtype == 'real':
                    raise ValueError("Cannot sum along a real dimension: {}".format(repr(v)))

            # Fuse Gaussians along a plate. Compare to eager_add_gaussian_gaussian().
            old_ints = OrderedDict((k, v) for k, v in self.inputs.items() if v.dtype != 'real')
            new_ints = OrderedDict((k, v) for k, v in old_ints.items() if k not in reduced_vars)
            inputs = OrderedDict((k, v) for k, v in self.inputs.items() if k not in reduced_vars)

            info_vec = Tensor(self.info_vec, old_ints).reduce(ops.add, reduced_vars)
            precision = Tensor(self.precision, old_ints).reduce(ops.add, reduced_vars)
            assert info_vec.inputs == new_ints
            assert precision.inputs == new_ints
            return Gaussian(info_vec.data, precision.data, inputs)

        return None  # defer to default implementation

    def unscaled_sample(self, sampled_vars, sample_inputs, rng_key=None):
        sampled_vars = sampled_vars.intersection(self.inputs)
        if not sampled_vars:
            return self
        if any(self.inputs[k].dtype != 'real' for k in sampled_vars):
            raise ValueError('Sampling from non-normalized Gaussian mixtures is intentionally '
                             'not implemented. You probably want to normalize. To work around, '
                             'add a zero Tensor/Array with given inputs.')

        # Partition inputs into sample_inputs + int_inputs + real_inputs.
        sample_inputs = OrderedDict((k, d) for k, d in sample_inputs.items()
                                    if k not in self.inputs)
        sample_shape = tuple(int(d.dtype) for d in sample_inputs.values())
        int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
        real_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype == 'real')
        inputs = sample_inputs.copy()
        inputs.update(int_inputs)

        if sampled_vars == frozenset(real_inputs):
            shape = sample_shape + self.info_vec.shape
            backend = get_backend()
            if backend != "numpy":
                from importlib import import_module
                dist = import_module(funsor.distribution.BACKEND_TO_DISTRIBUTIONS_BACKEND[backend])
                sample_args = (shape,) if rng_key is None else (rng_key, shape)
                white_noise = dist.Normal.dist_class(0, 1).sample(*sample_args)
            else:
                white_noise = np.random.randn(*shape)
            white_noise = ops.unsqueeze(white_noise, -1)

            white_vec = ops.triangular_solve(self.info_vec[..., None], self._precision_chol)
            sample = ops.triangular_solve(white_noise + white_vec, self._precision_chol, transpose=True)[..., 0]
            offsets, _ = _compute_offsets(real_inputs)
            results = []
            for key, domain in real_inputs.items():
                data = sample[..., offsets[key]: offsets[key] + domain.num_elements]
                data = data.reshape(shape[:-1] + domain.shape)
                point = Tensor(data, inputs)
                assert point.output == domain
                results.append(Delta(key, point))
            results.append(self.log_normalizer)
            return reduce(ops.add, results)

        raise NotImplementedError('TODO implement partial sampling of real variables')


@eager.register(Binary, AddOp, Gaussian, Gaussian)
def eager_add_gaussian_gaussian(op, lhs, rhs):
    # Fuse two Gaussians by adding their log-densities pointwise.
    # This is similar to a Kalman filter update, but also keeps track of
    # the marginal likelihood which accumulates into a Tensor.

    # Align data.
    inputs = lhs.inputs.copy()
    inputs.update(rhs.inputs)
    lhs_info_vec, lhs_precision = align_gaussian(inputs, lhs)
    rhs_info_vec, rhs_precision = align_gaussian(inputs, rhs)

    # Fuse aligned Gaussians.
    info_vec = lhs_info_vec + rhs_info_vec
    precision = lhs_precision + rhs_precision
    return Gaussian(info_vec, precision, inputs)


@eager.register(Binary, SubOp, Gaussian, (Funsor, Align, Gaussian))
@eager.register(Binary, SubOp, (Funsor, Align, Delta), Gaussian)
def eager_sub(op, lhs, rhs):
    return lhs + -rhs


@eager.register(Unary, NegOp, Gaussian)
def eager_neg(op, arg):
    info_vec = -arg.info_vec
    precision = -arg.precision
    return Gaussian(info_vec, precision, arg.inputs)


__all__ = [
    'BlockMatrix',
    'BlockVector',
    'Gaussian',
    'align_gaussian',
]
