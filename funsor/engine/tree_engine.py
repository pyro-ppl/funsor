from __future__ import absolute_import, division, print_function

from collections import defaultdict

from six.moves import reduce

import funsor.ops as ops
from funsor.distributions import Distribution
from funsor.engine.materialize import materialize
from funsor.pattern import match_commutative, try_match_reduction
from funsor.terms import Funsor


class Contractor(object):
    def __init__(self, reduce_dims):
        self.reduce_dims = reduce_dims
        self.latents = {}  # dim -> distribution
        self.downstream = defaultdict(set)  # dim -> set of downstream dims
        self.observations = defaultdict(list)  # : dim -> list of observation distributions
        self.results = []  # funsors

    def add(self, x):
        if not self.reduce_dims.intersection(x.dims):
            # This is effectively a scalar for present purposes.
            self.results.append(x)
        elif isinstance(x, Distribution):
            dims = self.reduce_dims.intersection(x.dims)
            if self.reduce_dims.isdisjoint(x.value.dims):
                # This is an observation.
                if len(dims) != 1:
                    raise ValueError(x)
                dim = next(iter(dims))
                self.observations[dim].append(x)
            else:
                # This is a sample site.
                if len(x.value.dims) != 1:
                    raise ValueError(x)
                dim = next(iter(x.value.dims))
                self.latents[dim] = x
                self.downstream[dim]
                self.observations[dim]
                for parent_dim in self.reduce_dims.intersection(x.dims):
                    if parent_dim != dim:
                        self.downstream[parent_dim].add(dim)
        else:
            raise ValueError(x)

    def pop_latent(self):
        for dim, downstream in self.downstream.items():
            if not downstream:
                del self.downstream[dim]
                latent = self.latents.pop(dim)
                observations = self.observations[dim]
                for parent_dim in self.reduce_dims.intersection(latent.dims):
                    if parent_dim != dim:
                        self.downstream[parent_dim].remove(dim)
                return dim, latent, observations
        raise ValueError


def _contract(operands, reduce_dims, default_cost=10):
    reduce_dims = frozenset(reduce_dims)
    c = Contractor(reduce_dims)
    for x in match_commutative(ops.add, *operands):
        c.add(x)
    while c.latents:
        dim, latent, observations = c.pop_latent()
        print('LATENT {}\n  {}'.format(dim, latent))
        for obs in observations:
            print('  OBS {}'.format(obs))
            latents = []
            for x in match_commutative(ops.add, latent + obs):
                if dim not in x.dims:
                    c.add(x)
                else:
                    latents.append(x)
            assert len(latents) == 1, latents
            latent = latents[0]
        c.add(latent.reduce(ops.logaddexp, dim))
    return reduce(ops.add, c.results)


def eval(x):
    """
    Restricted evaluator for log-sum-product-exp contractions on tree-shaped
    directed graphical models.
    """
    assert isinstance(x, Funsor)

    # Handle log-sum-product-exp contractions.
    for arg, reduce_dims in try_match_reduction(ops.logaddexp, x):
        operands = match_commutative(ops.add, arg)
        operands = tuple(materialize(x) for x in operands)
        return _contract(operands, reduce_dims)

    return materialize(x)


__all__ = [
    'eval',
]
