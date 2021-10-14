# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from . import ops
from .domains import Reals
from .gaussian import Gaussian
from .interpretations import StatefulInterpretation
from .terms import Approximate, Funsor, Variable


class Precondition(StatefulInterpretation):
    """
    Preconditioning interpretation for adjoint computations.

    This interpretation is intended to be used once, followed by a call to
    :meth:`combine_subs` as follows::

        # Lazily build a factor graph.
        with reflect:
            log_joint = Gaussian(...) + ... + Gaussian(...)
            log_Z = log_joint.reduce(ops.logaddexp)

        # Run a backward sampling under the precondition interpretation.
        with Precondition() as p:
            marginals = adjoint(
                ops.logaddexp, ops.add, log_Z, batch_vars=p.sample_vars
            )
        combine_subs = p.combine_subs()

        # Extract samples from Delta distributions.
        samples = {
            k: v(**combine_subs)
            for name, delta in marginals.items()
            for k, v in funsor.montecarlo.extract_samples(delta).items()
        }

    See :func:`~funsor.recipes.forward_filter_backward_precondition` for
    complete usage.

    :param str aux_name: Name of the auxiliary variable containing white noise.
    """

    def __init__(self, aux_name="aux"):
        super().__init__("precondition")
        self.aux_name = aux_name
        self.sample_inputs = OrderedDict()
        self.sample_vars = set()

    def combine_subs(self):
        """
        Method to create a combining substitution after preconditioning is
        complete. The returned substitution replaces per-factor auxiliary
        variables with slices into a single combined auxiliary variable.

        :returns: A substitution indexing each factor-wise auxiliary variable
            into a single global auxiliary variable.
        :rtype: dict
        """
        total_size = sum(v.num_elements for v in self.sample_inputs.values())
        aux = Variable(self.aux_name, Reals[total_size])
        subs = {}
        start = 0
        for k, v in self.sample_inputs.items():
            stop = start + v.num_elements
            subs[k] = aux[start:stop].reshape(v.shape)
            start = stop
        return subs


@Precondition.register(Approximate, ops.LogaddexpOp, Funsor, Gaussian, frozenset)
def precondition_approximate(state, op, model, guide, approx_vars):
    # Eagerly winnow approx_vars.
    approx_vars = approx_vars.intersection(guide.input_vars)
    if not approx_vars:
        return model

    # Determine how much white noise is needed to generate a sample.
    batch_shape = []
    event_numel = 0
    for k, v in guide.inputs.items():
        if v.dtype == "real":
            event_numel += v.num_elements
        else:
            batch_shape += v.size
    shape = tuple(batch_shape) + (event_numel,)
    name = f"{state.aux_name}_{len(state.sample_inputs)}"
    state.sample_inputs[name] = Reals[shape]
    state.sample_vars.add(Variable(name, Reals[shape]))

    # Precondition this factor.
    sample = guide.sample(approx_vars, OrderedDict([(name, Reals[shape])]))
    assert sample is not guide, "no progress"

    result = sample + model - guide
    return result


__all__ = [
    "Precondition",
]
