from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pyro.distributions as dist
import torch
from pyro.distributions.util import broadcast_shape

from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.interpreter import interpretation, reinterpret
from funsor.joint import Joint
from funsor.optimizer import apply_optimizer
from funsor.terms import Funsor, Number, lazy
from funsor.torch import Tensor

# FunsorDistribution uses a standard naming of Pyro batch dims.
DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
NAME_TO_DIM = dict(zip(DIM_TO_NAME, range(-100, 0)))


def _requires_grad(f):
    assert isinstance(f, Funsor)
    if isinstance(f, Number):
        return False
    if isinstance(f, Tensor):
        return f.data.requires_grad
    raise ValueError("Unsupported type: {}".format(type(f)))


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
        value = self.tensor_to_funsor(value, event_dim=self.event_dim)
        with interpretation(lazy):
            log_prob = apply_optimizer(self.funsor_dist(value=value))
        log_prob = reinterpret(log_prob)
        log_prob = self.funsor_to_tensor(log_prob)
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
        assert not _requires_grad(delta.log_prob), "distribution is not fully reparametrized"
        value = delta.point
        return value

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("TODO")

    def tensor_to_funsor(self, tensor, event_dim=0, dtype="real"):
        assert isinstance(tensor, torch.Tensor)
        batch_shape = tensor.shape[:tensor.dim() - event_dim]
        event_shape = tensor.shape[tensor.dim() - event_dim:]

        # Squeeze batch_shape.
        inputs = OrderedDict()
        squeezed_batch_shape = []
        for dim, size in enumerate(batch_shape):
            if size > 1:
                name = DIM_TO_NAME[dim - len(batch_shape)]
                inputs[name] = bint(size)
                squeezed_batch_shape.append(size)
        squeezed_batch_shape = torch.Size(squeezed_batch_shape)
        if squeezed_batch_shape != batch_shape:
            batch_shape = squeezed_batch_shape
            tensor = tensor.reshape(batch_shape + event_shape)

        return Tensor(tensor, inputs, dtype)

    def funsor_to_tensor(self, funsor_):
        assert isinstance(funsor_, Tensor)
        assert all(k.startswith("_pyro_dim_") for k in funsor_.inputs)
        names = tuple(sorted(funsor_.inputs, key=NAME_TO_DIM.__getitem__))
        tensor = funsor_.align(names).data
        if names:
            # Unsqueeze batch_shape.
            dims = list(map(NAME_TO_DIM.__getitem__, names))
            batch_shape = [1] * (-dims[0])
            for dim, size in zip(dims, tensor.shape):
                batch_shape[dim] = size
            tensor = tensor.reshape(batch_shape + funsor_.output.shape)
        return tensor

    def dist_to_funsor(self, pyro_dist, reinterpreted_batch_ndims=0):
        assert isinstance(pyro_dist, torch.distributions.Distribution)
        while isinstance(pyro_dist, dist.Independent):
            reinterpreted_batch_ndims += pyro_dist.reinterpreted_batch_ndims
            pyro_dist = pyro_dist.base_dist
        event_dim = pyro_dist.event_dim + reinterpreted_batch_ndims
        if isinstance(pyro_dist, dist.Categorical):
            return self.tensor_to_funsor(pyro_dist.logits, event_dim=event_dim + 1)
        if isinstance(pyro_dist, dist.Normal):
            raise NotImplementedError("TODO")
        if isinstance(pyro_dist, dist.MultivariateNormal):
            raise NotImplementedError("TODO")

        # Fall back to lazy wrapper.
        batch_shape = pyro_dist.batch_shape
        inputs = OrderedDict([(NAME_TO_DIM[dim - len(batch_shape)], bint(size))
                              for dim, size in enumerate(batch_shape) if size > 1])
        return DistributionFunsor(self, pyro_dist, inputs)


class DistributionFunsor(Funsor):
    """
    :class:`~funsor.terms.Funsor` wrapper around a
    :class:`~torch.distributions.Distribution` for use in Funsor code.
    """

    def __init__(self, pyro_dist, inputs=None):
        if inputs is None:
            inputs = OrderedDict()
        output = reals()
        super(DistributionFunsor, self).__init__(self, inputs, output)

    def unscaled_sample(self, sampled_vars, sample_inputs):
        raise NotImplementedError("TODO")


class DiscreteDiscreteHMM(FunsorDistribution):
    def __init__(self, initial_logits, transition_logits, observation_logits):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(observation_logits, torch.Tensor)
        assert initial_logits.dim() >= 1
        assert transition_logits.dim() >= 2
        assert observation_logits.dim() >= 2
        time_shape = broadcast_shape((1,), transition_logits.shape[-3:-2],
                                     observation_logits.shape[-3:-2])
        time_domain = bint(time_shape[0])
        event_shape = time_shape + observation_logits.shape[-1:]
        batch_shape = broadcast_shape(initial_logits.shape[:-1],
                                      transition_logits.shape[:-3],
                                      observation_logits.shape[:-3])

        # Convert tensors to funsors.
        initial_logits = self.tensor_to_funsor(initial_logits, event_dim=1)

        inputs = self.inputs.copy()
        if transition_logits.dim() >= 3 and transition_logits.size(-3) > 1:
            inputs["time"] = time_domain
        transition_logits = self.tensor_to_funsor(transition_logits, event_dim=2, inputs=inputs)

        inputs = self.inputs.copy()
        if observation_logits.dim() >= 3 and observation_logits.size(-3) > 1:
            inputs["time"] = time_domain
        observation_logits = self.tensor_to_funsor(observation_logits, event_dim=2, inputs=inputs)

        with interpretation(lazy):
            funsor_dist = initial_logits + transition_logits + observation_logits
        super(DiscreteDiscreteHMM, self).__init__(funsor_dist, batch_shape, event_shape)


class DiscreteHMM(FunsorDistribution):
    def __init__(self, initial_logits, transition_logits, observation_dist):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(observation_dist, torch.distributions.Distribution)
        assert initial_logits.dim() >= 1
        assert transition_logits.dim() >= 2
        assert len(observation_dist.batch_shape) >= 1
        time_shape = broadcast_shape((1,), transition_logits.shape[-3:-2],
                                     observation_dist.batch_shape[-2:-1])
        time_domain = bint(time_shape[0])
        event_shape = time_shape + observation_dist.event_shape
        batch_shape = broadcast_shape(initial_logits.shape[:-1],
                                      transition_logits.shape[:-3],
                                      observation_dist.batch_shape[:-2])
        self._has_rsample = observation_dist.has_rsample

        # Convert tensors and distributions to funsors.
        initial_logits = self.tensor_to_funsor(initial_logits, event_dim=1)

        inputs = self.inputs.copy()
        if transition_logits.dim() >= 3 and transition_logits.size(-3) > 1:
            inputs["time"] = time_domain
        transition_logits = self.tensor_to_funsor(transition_logits, event_dim=2, inputs=inputs)

        inputs = self.inputs.copy()
        if len(observation_dist.batch_shape) >= 2 and observation_dist.batch_shape[-2] > 1:
            inputs["time"] = time_domain
        observation_dist = self.dist_to_funsor(observation_dist, inputs=inputs)

        with interpretation(lazy):
            funsor_dist = initial_logits + transition_logits + observation_dist
        super(DiscreteHMM, self).__init__(funsor_dist, batch_shape, event_shape)

    @torch.distributions.constraints.dependent_property
    def has_rsample(self):
        return self._has_rsample


class GaussianGaussianHMM(FunsorDistribution):
    has_rsample = True

    def __init__(self, initial_dist, transition_matrix, transition_dist,
                 observation_matrix, observation_dist):
        assert isinstance(initial_dist, dist.MultivariateNormal)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_dist, dist.MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_dist, dist.MultivariateNormal)
        # TODO assert shapes...
        raise NotImplementedError("TODO")
