# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Recipes using Funsor
--------------------
This module provides a number of high-level algorithms using Funsor.

"""

from typing import Dict, FrozenSet

import funsor


def forward_filter_backward_rsample(
    factors: Dict[str, funsor.Funsor],
    eliminate: FrozenSet[str],
    plates: FrozenSet[str],
    sample_inputs: Dict[str, funsor.domains.Domain] = {},
):
    """
    A forward-filter backward-batched-reparametrized-sample algorithm for use
    in variational inference. The motivating use case is performing Gaussian
    tensor variable elimination over structured variational posteriors.

    :param dict factors: A dictionary mapping sample site name to a Funsor
        factor created at that sample site.
    :param frozenset: A set of names of latent variables to marginalize and
        plates to aggregate.
    :param plates: A set of names of plates to aggregate.
    :param dict sample_inputs: An optional dict of enclosing sample indices
        over which samples will be drawn in batch.
    :returns: A pair ``samples:Dict[str, Tensor], log_prob: Tensor`` of samples
        and log density evaluated at each of those samples. If ``sample_inputs``
        is nonempty, both outputs will be batched.
    :rtype: tuple
    """
    # Perform tensor variable elimination.
    with funsor.interpretations.reflect:
        log_Z = funsor.sum_product.sum_product(
            funsor.ops.logaddexp,
            funsor.ops.add,
            list(factors.values()),
            eliminate,
            plates,
        )
        log_Z = funsor.optimizer.apply_optimizer(log_Z)
    with funsor.montecarlo.MonteCarlo(**sample_inputs):
        log_Z, marginals = funsor.adjoint.forward_backward(
            funsor.ops.logaddexp, funsor.ops.add, log_Z
        )

    # Extract sample tensors.
    samples = {}
    for name, factor in factors.items():
        if name in eliminate:
            samples.update(funsor.montecarlo.extract_samples(marginals[factor]))
    assert frozenset(samples) == eliminate - plates

    # Compute log density at each sample.
    log_prob = -log_Z
    for f in factors.values():
        term = f(**samples)
        plates = eliminate.intersection(term.inputs)
        term = term.reduce(funsor.ops.add, plates)
        log_prob += term
    assert set(log_prob.inputs) == set(sample_inputs)

    return samples, log_prob
