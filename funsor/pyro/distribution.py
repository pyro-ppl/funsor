from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pyro.distributions as dist
import torch

from funsor.delta import Delta
from funsor.interpreter import interpretation, reinterpret
from funsor.joint import Joint
from funsor.optimizer import apply_optimizer
from funsor.pyro.convert import funsor_to_tensor, tensor_to_funsor
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
    def __init__(self, funsor_dist, batch_shape=torch.Size(), event_shape=torch.Size()):
        assert isinstance(funsor_dist, Funsor)
        assert "value" in funsor_dist.inputs
        super(FunsorDistribution, self).__init__(batch_shape, event_shape)
        self.funsor_dist = funsor_dist

    def log_prob(self, value):
        value = tensor_to_funsor(value, event_dim=self.event_dim)
        with interpretation(lazy):
            log_prob = apply_optimizer(self.funsor_dist(value=value))
        log_prob = reinterpret(log_prob)
        log_prob = funsor_to_tensor(log_prob)
        return log_prob

    def _sample_delta(self, sample_shape):
        sample_inputs = None
        if sample_shape:
            sample_inputs = OrderedDict("TODO")
        delta = self.funsor_dist.sample(frozenset({"value"}), sample_inputs)
        if isinstance(delta, Joint):
            delta, = delta.deltas
        assert isinstance(delta, Delta)
        return delta

    def sample(self, sample_shape=torch.Size()):
        delta = self._sample_delta(sample_shape)
        value = delta.point
        return value.detach()

    def rsample(self, sample_shape=torch.Size()):
        delta = self._sample_delta(sample_shape)
        assert not delta.log_prob.requires_grad, "distribution is not fully reparametrized"
        value = delta.point
        return value

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("TODO")
