import math
import warnings
from collections import OrderedDict, defaultdict
from functools import reduce

import torch
from pyro.distributions.util import broadcast_shape

import funsor.ops as ops
import funsor.torch_patch  # noqa F401
from funsor.delta import Delta
from funsor.domains import reals
from funsor.integrate import Integrate, integrator
from funsor.ops import AddOp, NegOp, SubOp
from funsor.terms import Align, Binary, Funsor, FunsorMeta, Number, Subs, Unary, Variable, eager, reflect, to_funsor
from funsor.torch import Tensor, align_tensor, align_tensors, materialize
from funsor.util import lazy_property


def _log_det_tri(x):
    return x.diagonal(dim1=-1, dim2=-2).log().sum(-1)


def _det_tri(x):
    return x.diagonal(dim1=-1, dim2=-2).prod(-1)


def _vv(vec1, vec2):
    """
    Computes the inner product ``< vec1 | vec 2 >``.
    """
    return vec1.unsqueeze(-2).matmul(vec2.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def _mv(mat, vec):
    return torch.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)


def _vmv(mat, vec):
    """
    Computes the inner product ``<vec | mat | vec>``.
    """
    vt = vec.unsqueeze(-2)
    v = vec.unsqueeze(-1)
    result = torch.matmul(vt, torch.matmul(mat, v))
    return result.squeeze(-1).squeeze(-1)


def _trace_mm(x, y):
    """
    Computes ``trace(x @ y)``.
    """
    assert x.dim() >= 2
    assert y.dim() >= 2
    xy = x * y
    return xy.reshape(xy.shape[:-2] + (-1,)).sum(-1)


def _pinverse(mat):
    """
    Like torch.pinverse() but supports batching.
    """
    shape = mat.shape
    mat = mat.reshape((-1,) + mat.shape[-2:])
    if mat.size(0) == 1:
        flat = mat[0].pinverse()
    else:
        flat = torch.stack([m.pinverse() for m in mat])
    return flat.reshape(shape)


def sym_inverse(mat):
    r"""
    Computes ``inverse(mat)`` assuming mat is symmetric and usually positive
    definite, but falling back to general pseudoinverse if positive
    definiteness fails.
    """
    try:
        # Attempt to use stable positive definite math.
        return mat.cholesky().cholesky_inverse()
    except RuntimeError as e:
        warnings.warn(e, RuntimeWarning)

    # Try masked reciprocal.
    if mat.size(-1) == 1:
        result = mat.reciprocal()
        result[(mat != 0) == 0] = 0
        return result

    # Fall back to pseudoinverse.
    return _pinverse(mat)


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


def _find_gaps(intervals, end):
    intervals = list(sorted(intervals))
    stops = [0] + [stop for start, stop in intervals]
    starts = [start for start, stop in intervals] + [end]
    return [(stop, start) for stop, start in zip(stops, starts) if stop != start]


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
            value = value.unsqueeze(pos - len(index))
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
        options = dict(dtype=prototype.dtype, device=prototype.device)
        for i in _find_gaps(self.parts.keys(), self.shape[-1]):
            self.parts[i] = torch.zeros(self.shape[:-1] + (i[1] - i[0],), **options)

        # Concatenate parts.
        parts = [v for k, v in sorted(self.parts.items())]
        result = torch.cat(parts, dim=-1)
        if not torch._C._get_tracing_state():
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
        options = dict(dtype=prototype.dtype, device=prototype.device)
        i_gaps = _find_gaps(self.parts.keys(), self.shape[-2])
        j_gaps = _find_gaps(arbitrary_row.keys(), self.shape[-1])
        rows = set().union(i_gaps, self.parts)
        cols = set().union(j_gaps, arbitrary_row)
        for i in rows:
            for j in cols:
                if j not in self.parts[i]:
                    shape = self.shape[:-2] + (i[1] - i[0], j[1] - j[0])
                    self.parts[i][j] = torch.zeros(shape, **options)

        # Concatenate parts.
        # TODO This could be optimized into a single .reshape().cat().reshape() if
        #   all inputs are contiguous, thereby saving a memcopy.
        columns = {i: torch.cat([v for j, v in sorted(part.items())], dim=-1)
                   for i, part in self.parts.items()}
        result = torch.cat([v for i, v in sorted(columns.items())], dim=-2)
        if not torch._C._get_tracing_state():
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
        old_loc = info_vec
        old_precision = precision
        info_vec = BlockVector(old_loc.shape[:-1] + (new_dim,))
        precision = BlockMatrix(old_loc.shape[:-1] + (new_dim, new_dim))
        for k1, new_offset1 in new_offsets.items():
            if k1 not in old_offsets:
                continue
            offset1 = old_offsets[k1]
            num_elements1 = old.inputs[k1].num_elements
            old_slice1 = slice(offset1, offset1 + num_elements1)
            new_slice1 = slice(new_offset1, new_offset1 + num_elements1)
            info_vec[..., new_slice1] = old_loc[..., old_slice1]
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

    Note that :class:`Gaussian` s are not normalized, rather they are
    canonicalized to evaluate to zero log density at the origin. This canonical
    form is useful in combination with the information filter representation
    because it allows :class:`Gaussian` s with incomplete information, i.e.
    zero eigenvalues in the precision matrix.  These incomplete distributions
    arise when making low-dimensional observations on higher dimensional hidden
    state.

    :param torch.Tensor info_vec: An optional batched information vector,
        where ``info_vec = precision @ mean``.
    :param torch.Tensor precision: A batched positive semidefinite precision
        matrix.
    :param OrderedDict inputs: Mapping from name to
        :class:`~funsor.domains.Domain` .
    """
    def __init__(self, info_vec, precision, inputs):
        assert isinstance(info_vec, torch.Tensor)
        assert isinstance(precision, torch.Tensor)
        assert isinstance(inputs, tuple)
        inputs = OrderedDict(inputs)

        # Compute total dimension of all real inputs.
        dim = sum(d.num_elements for d in inputs.values() if d.dtype == 'real')
        if not torch._C._get_tracing_state():
            assert dim
            assert precision.dim() >= 2 and precision.shape[-2:] == (dim, dim)
            assert info_vec.dim() >= 1 and info_vec.size(-1) == dim

        # Compute total shape of all bint inputs.
        batch_shape = tuple(d.dtype for d in inputs.values()
                            if isinstance(d.dtype, int))
        if not torch._C._get_tracing_state():
            assert precision.shape == batch_shape + (dim, dim)
            assert info_vec.shape == batch_shape + (dim,)

        output = reals()
        fresh = frozenset(inputs.keys())
        bound = frozenset()
        super(Gaussian, self).__init__(inputs, output, fresh, bound)
        self.info_vec = info_vec
        self.precision = precision
        self.batch_shape = batch_shape
        self.event_shape = (dim,)

    @lazy_property
    def _precision_chol(self):
        return self.precision.cholesky()

    @lazy_property
    def _log_normalizer(self):
        dim = self.precision.size(-1)
        log_det_term = _log_det_tri(self._precision_chol)
        data = 0.5 * math.log(2 * math.pi) * dim - log_det_term
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
        subs = tuple((k, materialize(to_funsor(v, self.inputs[k])))
                     for k, v in subs if k in self.inputs)
        if not subs:
            return self

        # Constants and Variables are eagerly substituted;
        # everything else is lazily substituted.
        lazy_subs = tuple((k, v) for k, v in subs
                          if not isinstance(v, (Number, Tensor, Variable)))
        var_subs = tuple((k, v) for k, v in subs if isinstance(v, Variable))
        int_subs = tuple((k, v) for k, v in subs if isinstance(v, (Number, Tensor))
                         if v.dtype != 'real')
        real_subs = tuple((k, v) for k, v in subs if isinstance(v, (Number, Tensor))
                          if v.dtype == 'real')
        if not (var_subs or int_subs or real_subs):
            return reflect(Subs, self, lazy_subs)

        # First perform any variable substitutions.
        if var_subs:
            rename = {k: v.name for k, v in var_subs}
            inputs = OrderedDict((rename.get(k, k), d) for k, d in self.inputs.items())
            if len(inputs) != len(self.inputs):
                raise ValueError("Variable substitution name conflict")
            var_result = Gaussian(self.info_vec, self.precision, inputs)
            return Subs(var_result, int_subs + real_subs + lazy_subs)

        # Next perform any integer substitution, i.e. slicing into a batch.
        if int_subs:
            int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
            real_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype == 'real')
            tensors = [self.info_vec, self.precision]
            funsors = [Subs(Tensor(x, int_inputs), int_subs) for x in tensors]
            inputs = funsors[0].inputs.copy()
            inputs.update(real_inputs)
            int_result = Gaussian(funsors[0].data, funsors[1].data, inputs)
            return Subs(int_result, real_subs + lazy_subs)

        # Broadcast all component tensors.
        real_subs = OrderedDict(subs)
        assert real_subs and not int_subs
        int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
        tensors = [Tensor(self.info_vec, int_inputs),
                   Tensor(self.precision, int_inputs)]
        tensors.extend(real_subs.values())
        int_inputs, tensors = align_tensors(*tensors)
        batch_dim = tensors[0].dim() - 1
        batch_shape = broadcast_shape(*(x.shape[:batch_dim] for x in tensors))
        (info_vec, precision), values = tensors[:2], tensors[2:]
        offsets, event_size = _compute_offsets(self.inputs)
        slices = [(k, slice(offset, offset + self.inputs[k].num_elements))
                  for k, offset in offsets.items()]

        # Expand all substituted values.
        values = OrderedDict(zip(real_subs, values))
        for k, value in values.items():
            value = value.reshape(value.shape[:batch_dim] + (-1,))
            if not torch._C._get_tracing_state():
                assert value.size(-1) == self.inputs[k].num_elements
            values[k] = value.expand(batch_shape + value.shape[-1:])

        # Try to perform a complete substitution of all real variables, resulting in a Tensor.
        if all(k in real_subs for k, d in self.inputs.items() if d.dtype == 'real'):
            # Form the concatenated value.
            value = BlockVector(batch_shape + (event_size,))
            for k, i in slices:
                if k in values:
                    value[..., i] = values[k]
            value = value.as_tensor()

            # Evaluate the non-normalized log density.
            result = (info_vec * value).sum(-1) - 0.5 * _vmv(precision, value)

            result = Tensor(result, int_inputs)
            assert result.output == reals()
            return Subs(result, lazy_subs)

        # Perform a partial substution of a subset of real variables, resulting in a Joint.
        # We split real inputs into two sets: a for the preserved and b for the substituted.
        b = frozenset(k for k, v in real_subs.items())
        a = frozenset(k for k, d in self.inputs.items() if d.dtype == 'real' and k not in b)
        prec_aa = torch.cat([torch.cat([
            precision[..., i1, i2]
            for k2, i2 in slices if k2 in a], dim=-1)
            for k1, i1 in slices if k1 in a], dim=-2)
        prec_ab = torch.cat([torch.cat([
            precision[..., i1, i2]
            for k2, i2 in slices if k2 in b], dim=-1)
            for k1, i1 in slices if k1 in a], dim=-2)
        prec_bb = torch.cat([torch.cat([
            precision[..., i1, i2]
            for k2, i2 in slices if k2 in b], dim=-1)
            for k1, i1 in slices if k1 in b], dim=-2)
        info_a = torch.cat([info_vec[..., i] for k, i in slices if k in a], dim=-1)
        info_b = torch.cat([info_vec[..., i] for k, i in slices if k in b], dim=-1)
        value_b = torch.cat([values[k] for k, i in slices if k in b], dim=-1)
        info_vec = info_a - _mv(prec_ab, value_b)
        log_scale = (info_b * value_b).sum(-1) - 0.5 * _vmv(prec_bb, value_b)
        precision = prec_aa.expand(info_vec.shape + (-1,))
        inputs = int_inputs.copy()
        for k, d in self.inputs.items():
            if k not in real_subs:
                inputs[k] = d
        return Gaussian(info_vec, precision, inputs) + Tensor(log_scale, int_inputs)

    def eager_reduce(self, op, reduced_vars):
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
                result = self._log_normalizer
            else:
                int_inputs = OrderedDict((k, v) for k, v in inputs.items() if v.dtype != 'real')
                offsets, _ = _compute_offsets(self.inputs)
                a = []
                b = []
                for key, domain in self.inputs.items():
                    if domain.dtype == 'real':
                        block = range(offsets[key], offsets[key] + domain.num_elements)
                        (b if key in reduced_vars else a).extend(block)
                a = torch.tensor(a)
                b = torch.tensor(b)

                prec_aa = self.precision[..., a.unsqueeze(-1), a]
                prec_ba = self.precision[..., b.unsqueeze(-1), a]
                prec_bb = self.precision[..., b.unsqueeze(-1), b]
                prec_b = prec_bb.cholesky()
                prec_a = prec_ba.cholesky_solve(prec_b)
                prec_at = prec_a.transpose(-1, -2)
                precision = prec_aa - prec_at.matmul(prec_a)

                info_a = self.info_vec[..., a]
                info_b = self.info_vec[..., b]
                b_tmp = info_b.unsqueeze(-1).cholesky_solve(prec_b)
                info_vec = info_a - prec_at.matmul(b_tmp).squeeze(-1)

                log_prob = Tensor(0.5 * len(b) * math.log(2 * math.pi) + _log_det_tri(prec_b) +
                                  0.5 * b_tmp.squeeze(-1).pow(2).sum(-1),
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

    def unscaled_sample(self, sampled_vars, sample_inputs):
        # Sample only the real variables.
        sampled_vars = frozenset(k for k, v in self.inputs.items()
                                 if k in sampled_vars if v.dtype == 'real')
        if not sampled_vars:
            return self

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
            white_noise = torch.randn(shape + (1,))
            white_vec = self.info_vec.unsqueeze(-1).cholesky_solve(self._precision_chol)
            sample = (white_noise + white_vec).cholesky_solve(self._precision_chol).squeeze(-1)
            if not torch._C._get_tracing_state():
                assert sample.shape == self.info_vec.shape
            offsets, _ = _compute_offsets(real_inputs)
            results = []
            for key, domain in real_inputs.items():
                data = sample[..., offsets[key]: offsets[key] + domain.num_elements]
                data = data.reshape(shape[:-1] + domain.shape)
                point = Tensor(data, inputs)
                assert point.output == domain
                results.append(Delta(key, point))
            results.append(self._log_normalizer)
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
@eager.register(Binary, SubOp, (Funsor, Align), Gaussian)
def eager_sub(op, lhs, rhs):
    return lhs + -rhs


@eager.register(Unary, NegOp, Gaussian)
def eager_neg(op, arg):
    info_vec = -arg.info_vec
    precision = -arg.precision
    return Gaussian(info_vec, precision, arg.inputs)


@eager.register(Integrate, Gaussian, Variable, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if real_vars:
        assert real_vars == frozenset([integrand.name])
        data = log_measure.loc * log_measure._log_normalizer.data.exp().unsqueeze(-1)
        data = data.reshape(log_measure.loc.shape[:-1] + integrand.output.shape)
        inputs = OrderedDict((k, d) for k, d in log_measure.inputs.items() if d.dtype != 'real')
        return Tensor(data, inputs)

    return None  # defer to default implementation


@eager.register(Integrate, Gaussian, Gaussian, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if real_vars:

        lhs_reals = frozenset(k for k, d in log_measure.inputs.items() if d.dtype == 'real')
        rhs_reals = frozenset(k for k, d in integrand.inputs.items() if d.dtype == 'real')
        if lhs_reals == real_vars and rhs_reals <= real_vars:
            inputs = OrderedDict((k, d) for t in (log_measure, integrand)
                                 for k, d in t.inputs.items())
            lhs_loc, lhs_precision = align_gaussian(inputs, log_measure)
            rhs_loc, rhs_precision = align_gaussian(inputs, integrand)

            # Compute the expectation of a non-normalized quadratic form.
            # See "The Matrix Cookbook" (November 15, 2012) ss. 8.2.2 eq. 380.
            # http://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
            lhs_precision_chol = lhs_precision.cholesky()
            lhs_covariance = lhs_precision_chol.cholesky_inverse()
            dim = lhs_loc.size(-1)
            norm = (2 * math.pi) ** (0.5 * dim) / _det_tri(lhs_precision_chol)
            data = (-0.5) * norm * (_vmv(rhs_precision, lhs_loc - rhs_loc) +
                                    _trace_mm(rhs_precision, lhs_covariance))
            inputs = OrderedDict((k, d) for k, d in inputs.items() if k not in reduced_vars)
            result = Tensor(data, inputs)
            return result.reduce(ops.add, reduced_vars - real_vars)

        raise NotImplementedError('TODO implement partial integration')

    return None  # defer to default implementation


__all__ = [
    'BlockMatrix',
    'BlockVector',
    'Gaussian',
    'align_gaussian',
]
