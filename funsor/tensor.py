from collections import OrderedDict
from functools import reduce

import numpy as np
#import torch
from multipledispatch import dispatch

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import Domain, find_domain, reals
from funsor.terms import Funsor, FunsorMeta, Number, Slice, Variable


numeric_array = (torch.Tensor, np.ndarray)


class TensorMeta(FunsorMeta):
    """
    Wrapper to fill in default args and convert between OrderedDict and tuple.
    """
    def __call__(cls, data, inputs=None, dtype="real"):
        if inputs is None:
            inputs = tuple()
        elif isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        return super(TensorMeta, cls).__call__(data, inputs, dtype)


class Tensor(Funsor, metaclass=TensorMeta):
    """
    Funsor backed by a PyTorch Tensor.

    This follows the :mod:`torch.distributions` convention of arranging
    named "batch" dimensions on the left and remaining "event" dimensions
    on the right. The output shape is determined by all remaining dims.
    For example::

        data = torch.zeros(5,4,3,2)
        x = Tensor(data, OrderedDict([("i", bint(5)), ("j", bint(4))]))
        assert x.output == reals(3, 2)

    Operators like ``matmul`` and ``.sum()`` operate only on the output shape,
    and will not change the named inputs.

    :param numeric_array data: A PyTorch tensor or NumPy ndarray.
    :param OrderedDict inputs: An optional mapping from input name (str) to
        datatype (:class:`~funsor.domains.Domain` ). Defaults to empty.
    :param dtype: optional output datatype. Defaults to "real".
    :type dtype: int or the string "real".
    """
    def __init__(self, data, inputs=None, dtype="real"):
        assert isinstance(data, numeric_array)
        assert isinstance(inputs, tuple)
        if not torch._C._get_tracing_state():
            assert len(inputs) <= data.dim()
            for (k, d), size in zip(inputs, data.shape):
                assert d.dtype == size
        inputs = OrderedDict(inputs)
        output = Domain(data.shape[len(inputs):], dtype)
        fresh = frozenset(inputs.keys())
        bound = frozenset()
        super(Tensor, self).__init__(inputs, output, fresh, bound)
        self.data = data

    def __repr__(self):
        if self.output != "real":
            return 'Tensor({}, {}, {})'.format(self.data, self.inputs, repr(self.dtype))
        elif self.inputs:
            return 'Tensor({}, {})'.format(self.data, self.inputs)
        else:
            return 'Tensor({})'.format(self.data)

    def __str__(self):
        if self.dtype != "real":
            return 'Tensor({}, {}, {})'.format(self.data, self.inputs, repr(self.dtype))
        elif self.inputs:
            return 'Tensor({}, {})'.format(self.data, self.inputs)
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

    def clamp_finite(self):
        finfo = ops.finfo(self.data)
        data = ops.clamp(self.data, min=finfo.min, max=finfo.max)
        return Tensor(data, self.inputs, self.dtype)

    @property
    def requires_grad(self):
        # NB: numpy does not have attribute requires_grad
        return getattr(self.data, "requires_grad", None)

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self

        inputs = OrderedDict((name, self.inputs[name]) for name in names)
        inputs.update(self.inputs)
        old_dims = tuple(self.inputs)
        new_dims = tuple(inputs)
        permutation = tuple(old_dims.index(d) for d in new_dims)
        permutation = permutation + tuple(range(len(permutation), len(permutation) + len(self.output.shape)))
        data = self.data.permute(permutation)
        return Tensor(data, inputs, self.dtype)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = OrderedDict((k, to_funsor(v, self.inputs[k]))
                           for k, v in subs if k in self.inputs)
        if not subs:
            return self

        # Handle renaming to enable cons hashing, and
        # handle slicing to avoid copying data.
        if any(isinstance(v, (Variable, Slice)) for v in subs.values()):
            slices = None
            inputs = OrderedDict()
            for i, (k, d) in enumerate(self.inputs.items()):
                if k in subs:
                    v = subs[k]
                    if isinstance(v, Variable):
                        del subs[k]
                        k = v.name
                    elif isinstance(v, Slice):
                        del subs[k]
                        k = v.name
                        d = v.inputs[v.name]
                        if slices is None:
                            slices = [slice(None)] * self.data.dim()
                        slices[i] = v.slice
                inputs[k] = d
            data = self.data[tuple(slices)] if slices else self.data
            result = Tensor(data, inputs, self.dtype)
            return result.eager_subs(tuple(subs.items()))

        # materialize after checking for renaming case
        subs = OrderedDict((k, materialize(v)) for k, v in subs.items())

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
                    assert isinstance(v, Tensor)
                    v = v.align(tuple(k2 for k2 in inputs if k2 in v.inputs))
                    assert isinstance(v, Tensor)
                    v_shape = [1] * total_size
                    for k2, size in zip(v.inputs, v.data.shape):
                        v_shape[new_dims[k2]] = size
                    index.append(v.data.reshape(tuple(v_shape)))
            else:
                # Construct a [:] slice for this preserved input.
                offset_from_right = -1 - new_dims[k]
                index.append(ops.new_arange(self.data, domain.dtype).reshape(
                    (-1,) + (1,) * offset_from_right))

        # Construct a [:] slice for the output.
        for i, size in enumerate(self.output.shape):
            offset_from_right = len(self.output.shape) - i - 1
            index.append(ops.new_arange(self.data, size).reshape(
                (-1,) + (1,) * offset_from_right))

        data = self.data[tuple(index)]
        return Tensor(data, inputs, self.dtype)

    def eager_unary(self, op):
        dtype = find_domain(op, self.output).dtype
        if op in REDUCE_OP_TO_NUMERIC_ARRAY:
            batch_dim = len(self.data.shape) - len(self.output.shape)
            data = self.data.reshape(self.data.shape[:batch_dim] + (-1,))
            data = REDUCE_OP_TO_NUMERIC_ARRAY[op](data, -1)
            return Tensor(data, self.inputs, dtype)
        return Tensor(op(self.data), self.inputs, dtype)

    def eager_reduce(self, op, reduced_vars):
        if op in REDUCE_OP_TO_NUMERIC_ARRAY:
            torch_op = REDUCE_OP_TO_NUMERIC_ARRAY[op]
            assert isinstance(reduced_vars, frozenset)
            self_vars = frozenset(self.inputs)
            reduced_vars = reduced_vars & self_vars
            if reduced_vars == self_vars and not self.output.shape:
                # Reduce all dims at once.
                return Tensor(torch_op(self.data), dtype=self.dtype)

            # Reduce one dim at a time.
            data = self.data
            offset = 0
            for k, domain in self.inputs.items():
                if k in reduced_vars:
                    assert not domain.shape
                    data = torch_op(data, dim=offset)
                else:
                    offset += 1
            inputs = OrderedDict((k, v) for k, v in self.inputs.items()
                                 if k not in reduced_vars)
            return Tensor(data, inputs, self.dtype)
        return super(Tensor, self).eager_reduce(op, reduced_vars)

    # TODO: support a `key` kwarg to get samples from numpyro distributions
    def unscaled_sample(self, sampled_vars, sample_inputs):
        assert self.output == reals()
        sampled_vars = sampled_vars.intersection(self.inputs)
        if not sampled_vars:
            return self

        # Partition inputs into sample_inputs + batch_inputs + event_inputs.
        sample_inputs = OrderedDict((k, d) for k, d in sample_inputs.items()
                                    if k not in self.inputs)
        sample_shape = tuple(int(d.dtype) for d in sample_inputs.values())
        batch_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if k not in sampled_vars)
        event_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if k in sampled_vars)
        be_inputs = batch_inputs.copy()
        be_inputs.update(event_inputs)
        sb_inputs = sample_inputs.copy()
        sb_inputs.update(batch_inputs)

        # Sample all variables in a single Categorical call.
        logits = align_tensor(be_inputs, self)
        batch_shape = logits.shape[:len(batch_inputs)]
        flat_logits = logits.reshape(batch_shape + (-1,))
        sample_shape = tuple(d.dtype for d in sample_inputs.values())
        flat_sample = torch.distributions.Categorical(logits=flat_logits).sample(sample_shape)
        assert flat_sample.shape == sample_shape + batch_shape
        results = []
        mod_sample = flat_sample
        for name, domain in reversed(list(event_inputs.items())):
            size = domain.dtype
            point = Tensor(mod_sample % size, sb_inputs, size)
            mod_sample = mod_sample / size
            results.append(Delta(name, point))

        # Account for the log normalizer factor.
        # Derivation: Let f be a nonnormalized distribution (a funsor), and
        #   consider operations in linear space (source code is in log space).
        #   Let x0 ~ f/|f| be a monte carlo sample from a normalized f/|f|.
        #                              f(x0) / |f|      # dice numerator
        #   Let g = delta(x=x0) |f| -----------------
        #                           detach(f(x0)/|f|)   # dice denominator
        #                       |detach(f)| f(x0)
        #         = delta(x=x0) -----------------  be a dice approximation of f.
        #                         detach(f(x0))
        #   Then g is an unbiased estimator of f in value and all derivatives.
        #   In the special case f = detach(f), we can simplify to
        #       g = delta(x=x0) |f|.
        if flat_logits.requires_grad:
            # Apply a dice factor to preserve differentiability.
            index = [ops.new_arange(n).reshape((n,) + (1,) * (flat_logits.dim() - i - 2))
                     for i, n in enumerate(flat_logits.shape[:-1])]
            index.append(flat_sample)
            log_prob = flat_logits[index]
            assert log_prob.shape == flat_sample.shape
            results.append(Tensor(flat_logits.detach().logsumexp(-1) +
                                  (log_prob - log_prob.detach()), sb_inputs))
        else:
            # This is the special case f = detach(f).
            results.append(Tensor(flat_logits.logsumexp(-1), batch_inputs))

        return reduce(ops.add, results)


@dispatch(numeric_array)
def to_funsor(x):
    return Tensor(x)


REDUCE_OP_TO_NUMERIC_ARRAY = {
    ops.add: ops.sum,
    ops.mul: ops.prod,
    ops.and_: ops.all,
    ops.or_: ops.any,
    ops.logaddexp: ops.logsumexp,
    ops.min: ops.min_,
    ops.max: ops.max_,
}
