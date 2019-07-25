from collections import OrderedDict

import pyro.distributions as dist
import torch

from funsor.delta import Delta
from funsor.domains import bint
from funsor.interpreter import interpretation, reinterpret
from funsor.joint import Joint
from funsor.optimizer import apply_optimizer
from funsor.pyro.convert import DIM_TO_NAME, funsor_to_tensor, tensor_to_funsor
from funsor.terms import Funsor, lazy


class FunsorDistribution(dist.TorchDistribution):
    """
    :class:`~torch.distributions.Distribution` wrapper around a
    :class:`~funsor.terms.Funsor` for use in Pyro code. This is typically used
    as a base class for specific funsor inference algorithms wrapped in a
    distribution interface.

    :param funsor.terms.Funsor funsor_dist: A funsor with an input named
        "value" that is treated as a random variable. The distribution should
        be normalized over "value".
    :param torch.Size batch_shape: The distribution's batch shape. This must
        be in the same order as the input of the ``funsor_dist``, but may
        contain extra dims of size 1.
    :param event_shape: The distribution's event shape.
    """
    arg_constraints = {}

    def __init__(self, funsor_dist, batch_shape=torch.Size(), event_shape=torch.Size(),
                 dtype="real"):
        assert isinstance(funsor_dist, Funsor)
        assert isinstance(batch_shape, tuple)
        assert isinstance(event_shape, tuple)
        assert "value" in funsor_dist.inputs
        super(FunsorDistribution, self).__init__(batch_shape, event_shape)
        self.funsor_dist = funsor_dist
        self.dtype = dtype

    def log_prob(self, value):
        ndims = max(len(self.batch_shape), value.dim() - self.event_dim)
        value = tensor_to_funsor(value, event_output=self.event_dim, dtype=self.dtype)
        with interpretation(lazy):
            log_prob = apply_optimizer(self.funsor_dist(value=value))
        log_prob = reinterpret(log_prob)
        log_prob = funsor_to_tensor(log_prob, ndims=ndims)
        return log_prob

    def _sample_delta(self, sample_shape):
        sample_inputs = None
        if sample_shape:
            sample_inputs = OrderedDict()
            shape = sample_shape + self.batch_shape
            for dim in range(-len(shape), -len(self.batch_shape)):
                if shape[dim] > 1:
                    sample_inputs[DIM_TO_NAME[dim]] = bint(shape[dim])
        delta = self.funsor_dist.sample(frozenset({"value"}), sample_inputs)
        if isinstance(delta, Joint):
            delta, = delta.deltas
        assert isinstance(delta, Delta)
        return delta

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        delta = self._sample_delta(sample_shape)
        ndims = len(sample_shape) + len(self.batch_shape) + len(self.event_shape)
        value = funsor_to_tensor(delta.point, ndims=ndims)
        return value.detach()

    def rsample(self, sample_shape=torch.Size()):
        delta = self._sample_delta(sample_shape)
        assert not delta.log_prob.requires_grad, "distribution is not fully reparametrized"
        ndims = len(sample_shape) + len(self.batch_shape) + len(self.event_shape)
        value = funsor_to_tensor(delta.point, ndims=ndims)
        return value

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("TODO")
