# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Real
from funsor.terms import Funsor, FunsorMeta, Unary, Variable, eager
from funsor.util import get_default_dtype


class ProvenanceMeta(FunsorMeta):
    """
    Wrapper to collect provenance information from the term.
    """

    def __call__(cls, term, provenance):
        while isinstance(term, Provenance):
            provenance |= term.provenance
            term = term.term

        return super(ProvenanceMeta, cls).__call__(term, provenance)


class Provenance(Funsor, metaclass=ProvenanceMeta):
    """
    Provenance funsor for tracking ``(name, (point, log_density))`` of random variables
    samples that were in the history of computations of the tracked term.

    **References**
    [1] David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind (2011)
        Nonstandard Interpretations of Probabilistic Programs for Efficient Inference
        http://papers.neurips.cc/paper/4309-nonstandard-interpretations-of-probabilistic-programs-for-efficient-inference.pdf

    :param funsor data: A funsor that depends on provenance samples.
    :param frozenset provenance: A set of tuples of the form
        ``(name, (point, log_density))``.
    """

    def __init__(self, term, provenance):
        assert isinstance(term, Funsor)
        assert isinstance(provenance, frozenset)
        inputs = OrderedDict()
        for name, (point, log_density) in provenance:
            assert isinstance(name, str)
            assert isinstance(point, Funsor)
            assert isinstance(log_density, Funsor)
            assert log_density.output == Real
            assert name not in inputs
            assert name not in point.inputs
            inputs.update({name: point.output})
            inputs.update(point.inputs)

        inputs.update(term.inputs)
        output = term.output
        fresh = frozenset(name for name, term in provenance)
        bound = {}
        super(Provenance, self).__init__(inputs, output, fresh, bound)
        self.term = term
        self.provenance = provenance

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = OrderedDict(subs)
        assert set(subs.keys()).issubset(self.fresh)
        new_provenance = frozenset()
        new_term = self.term
        for name, (point, log_density) in self.provenance:
            if name in subs:
                value = subs[name]
                assert value.output == point.output
                if isinstance(value, Variable):
                    new_provenance |= frozenset([(value.name, (point, log_density))])
                    continue

                dtype = get_default_dtype()
                is_equal = ops.astype((value == point).all(), dtype)
                new_term = new_term * (is_equal.log() + log_density).exp()
            else:
                new_provenance |= frozenset([(name, (point, log_density))])
        return Provenance(new_term, new_provenance) if new_provenance else new_term

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        result = self.term._sample(sampled_vars, sample_inputs, rng_key)
        return Provenance(result, self.provenance)


@eager.register(Unary, ops.UnaryOp, Provenance)
def eager_unary(op, arg):
    return Provenance(op(arg.term), arg.provenance)
