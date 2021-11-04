# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from funsor.delta import Delta
from funsor.interpretations import DispatchedInterpretation
from funsor.terms import Funsor, eager, reflect


class Importance(Funsor):
    """
    Importance sampling for approximating integrals wrt a set of variables.

    When the proposal distribution (guide) is Delta then the eager
    interpretation is ``Delta + log_importance_weight``.
    The user-facing interface is the :meth:`Funsor.approximate` method.

    :param Funsor model: An exact funsor depending on ``sampled_vars``.
    :param Funsor guide: A proposal distribution.
    :param frozenset approx_vars: A set of variables over which to approximate.
    """

    def __init__(self, model, guide, sampled_vars):
        assert isinstance(model, Funsor)
        assert isinstance(guide, Funsor)
        assert isinstance(sampled_vars, frozenset), sampled_vars
        inputs = model.inputs.copy()
        inputs.update(guide.inputs)
        output = model.output
        super().__init__(inputs, output)
        self.model = model
        self.guide = guide
        self.sampled_vars = sampled_vars

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if not reduced_vars:
            return self

        return self.model.reduce(op, reduced_vars)


@eager.register(Importance, Funsor, Delta)
def eager_importance(model, guide):
    # Delta + log_importance_weight
    return guide + model - guide


lazy_importance = DispatchedInterpretation("lazy_importance")
"""
Lazy interpretation of Importance with a Delta guide.
"""


@lazy_importance.register(Importance, Funsor, Delta)
def _lazy_importance(model, guide):
    return reflect.interpret(Importance, model, guide)
