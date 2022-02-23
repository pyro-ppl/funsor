# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Recipes using Funsor
--------------------
This module provides a number of high-level algorithms using Funsor.

"""

from typing import Dict, FrozenSet

import funsor  # Let's use fully qualified names in this file.


def forward_filter_backward_rsample(
    factors: Dict[str, funsor.Funsor],
    eliminate: FrozenSet[str],
    plates: FrozenSet[str],
    sample_inputs: Dict[str, funsor.domains.Domain] = {},
    rng_key=None,
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
    :param rng_key: A random number key for the JAX backend.
    :returns: A pair ``samples:Dict[str, Tensor], log_prob: Tensor`` of samples
        and log density evaluated at each of those samples. If ``sample_inputs``
        is nonempty, both outputs will be batched.
    :rtype: tuple
    """
    assert isinstance(factors, dict)
    assert all(isinstance(k, str) for k in factors)
    assert all(isinstance(v, funsor.Funsor) for v in factors.values())
    assert isinstance(eliminate, frozenset)
    assert all(isinstance(v, str) for v in eliminate)
    assert isinstance(plates, frozenset)
    assert all(isinstance(v, str) for v in plates)
    assert isinstance(sample_inputs, dict)
    assert all(isinstance(k, str) for k in sample_inputs)
    assert all(isinstance(v, funsor.domains.Domain) for v in sample_inputs.values())

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
    batch_vars = frozenset(funsor.Variable(k, v) for k, v in sample_inputs.items())
    with funsor.montecarlo.MonteCarlo(**sample_inputs, rng_key=rng_key):
        log_Z, marginals = funsor.adjoint.forward_backward(
            funsor.ops.logaddexp, funsor.ops.add, log_Z, batch_vars=batch_vars
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


def forward_filter_backward_precondition(
    factors: Dict[str, funsor.Funsor],
    eliminate: FrozenSet[str],
    plates: FrozenSet[str],
    aux_name: str = "aux",
):
    """
    A forward-filter backward-precondition algorithm for use in variational
    inference or preconditioning in Hamiltonian Monte Carlo. The motivating use
    case is performing Gaussian tensor variable elimination over structured
    variational posteriors, and optionally using the learned posterior to
    determine momentum in HMC.

    :param dict factors: A dictionary mapping sample site name to a Funsor
        factor created at that sample site.
    :param frozenset: A set of names of latent variables to marginalize and
        plates to aggregate.
    :param plates: A set of names of plates to aggregate.
    :param str aux_name: Name of the auxiliary variable containing white noise.
    :returns: A pair ``samples:Dict[str, Tensor], log_prob: Tensor`` of samples
        and log density evaluated at each of those samples. Both outputs depend
        on a vector named by ``aux_name``, e.g. ``aux: Reals[d]`` where ``d``
        is the total number of elements in eliminated variables.
    :rtype: tuple
    """
    assert isinstance(factors, dict)
    assert all(isinstance(k, str) for k in factors)
    assert all(isinstance(v, funsor.Funsor) for v in factors.values())
    assert isinstance(eliminate, frozenset)
    assert all(isinstance(v, str) for v in eliminate)
    assert isinstance(plates, frozenset)
    assert all(isinstance(v, str) for v in plates)
    assert isinstance(aux_name, str)
    assert not any(aux_name in f.inputs for f in factors.values())

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
    with funsor.precondition.Precondition(aux_name=aux_name) as precondition:
        log_Z, marginals = funsor.adjoint.forward_backward(
            funsor.ops.logaddexp,
            funsor.ops.add,
            log_Z,
            batch_vars=precondition.sample_vars,
        )

    # Extract sample tensors.
    samples = {}
    for name, factor in factors.items():
        if name in eliminate:
            samples.update(funsor.montecarlo.extract_samples(marginals[factor]))
    assert frozenset(samples) == eliminate - plates

    # Combine into a single auxiliary variable.
    subs = precondition.combine_subs()
    samples = {k: v(**subs) for k, v in samples.items()}

    # Compute log density at each sample, lazily dependent on aux_name.
    log_prob = -log_Z
    for f in factors.values():
        term = f(**samples)
        plates = eliminate.intersection(term.inputs)
        term = term.reduce(funsor.ops.add, plates)
        log_prob += term
    assert set(log_prob.inputs) == {aux_name}

    return samples, log_prob
