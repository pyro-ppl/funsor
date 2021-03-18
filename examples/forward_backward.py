# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import funsor.ops as ops
from funsor import Funsor, Tensor
from funsor.adjoint import AdjointTape
from funsor.domains import Bint
from funsor.testing import assert_close, random_tensor


def forward_algorithm(
    factors: List[Funsor],
    step: Dict[str, str],
) -> Tuple[Funsor, List[Funsor]]:
    """
    Calculate log marginal probability using the forward algorithm:
    Z = p(y[0:T])

    Transition and emission probabilities are given by init and trans factors:
    init = p(y[0], x[0])
    trans[t] = p(y[t], x[t] | x[t-1])

    Forward probabilities are computed inductively:
    alpha[t] = p(y[0:t], x[t])
    alpha[0] = init
    alpha[t+1] = alpha[t] @ trans[t+1]
    """
    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    reduce_vars = frozenset(drop)

    # base case
    alpha = factors[0]
    alphas = [alpha]
    # inductive steps
    for trans in factors[1:]:
        alpha = (alpha(**curr_to_drop) + trans(**prev_to_drop)).reduce(
            ops.logaddexp, reduce_vars
        )
        alphas.append(alpha)
    else:
        Z = alpha(**curr_to_drop).reduce(ops.logaddexp, reduce_vars)
    return Z


def forward_backward_algorithm(
    factors: List[Funsor],
    step: Dict[str, str],
) -> List[Tensor]:
    """
    Calculate marginal probabilities:
    p(x[t], x[t-1] | Y)
    """
    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    reduce_vars = frozenset(drop)

    # Base cases
    alpha = factors[0]  # alpha[0] = p(y[0], x[0])
    beta = Tensor(
        ops.full_like(alpha.data, 0.0), alpha(x_curr="x_prev").inputs
    )  # beta[T] = 1

    # Backward algorithm
    # beta[t] = p(y[t+1:T] | x[t])
    # beta[t] = trans[t+1] @ beta[t+1]
    betas = [beta]
    for trans in factors[:0:-1]:
        beta = (trans(**curr_to_drop) + beta(**prev_to_drop)).reduce(
            ops.logaddexp, reduce_vars
        )
        betas.append(beta)
    else:
        init = factors[0]
        Z = (init(**curr_to_drop) + beta(**prev_to_drop)).reduce(
            ops.logaddexp, reduce_vars
        )
    betas.reverse()

    # Forward algorithm & Marginal computations
    marginal = alpha + betas[0](**{"x_prev": "x_curr"}) - Z  # p(x[0] | Y)
    marginals = [marginal]
    # inductive steps
    for trans, beta in zip(factors[1:], betas[1:]):
        # alpha[t-1] * trans[t] = p(y[0:t], x[t-1], x[t])
        alpha_trans = alpha(**{"x_curr": "x_prev"}) + trans
        # alpha[t] = p(y[0:t], x[t])
        alpha = alpha_trans.reduce(ops.logaddexp, "x_prev")
        # alpha[t-1] * trans[t] * beta[t] / Z = p(x[t-1], x[t] | Y)
        marginal = alpha_trans + beta(**{"x_prev": "x_curr"}) - Z
        marginals.append(marginal)

    return marginals


def main(args):
    """
    Compute marginal probabilities p(x[t], x[t-1] | Y) for an HMM:

    x[0] -> ... -> x[t-1] -> x[t] -> ... -> x[T]
     |              |         |             |
     v              v         v             v
    y[0]           y[t-1]    y[t]           y[T]

    Z = p(y[0:T])
    alpha[t] = p(y[0:t], x[t])
    beta[t] = p(y[t+1:T] | x[t])
    trans[t] = p(y[t], x[t] | x[t-1])

    p(x[t], x[t-1] | Y) = alpha[t-1] * beta[t] * trans[t] / Z

    d Z / d trans[t] = alpha[t-1] * beta[t]

    **References:**

    [1] Jason Eisner (2016)
        "Inside-Outside and Forward-Backward Algorithms Are Just Backprop
        (Tutorial Paper)"
        https://www.cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf
    [2] Zhifei Li and Jason Eisner (2009)
        "First- and Second-Order Expectation Semirings
        with Applications to Minimum-Risk Training on Translation Forests"
        http://www.cs.jhu.edu/~zfli/pubs/semiring_translation_zhifei_emnlp09.pdf
    """

    # transition and emission probabilities
    init = random_tensor(OrderedDict([("x_curr", Bint[args.hidden_dim])]))
    factors = [init]
    for t in range(args.time_steps - 1):
        factors.append(
            random_tensor(
                OrderedDict(x_prev=Bint[args.hidden_dim], x_curr=Bint[args.hidden_dim])
            )
        )

    # Compute marginal probabilities using the forward-backward algorithm
    marginals = forward_backward_algorithm(factors, {"x_prev": "x_curr"})
    # Compute marginal probabilities using backpropagation
    with AdjointTape() as tape:
        Z = forward_algorithm(factors, {"x_prev": "x_curr"})
    result = tape.adjoint(ops.logaddexp, ops.add, Z, factors)
    adjoint_terms = list(result.values())

    t = 0
    for expected, adj, trans in zip(marginals, adjoint_terms, factors):
        # adjoint returns dZ / dtrans = alpha[t-1] * beta[t]
        # marginal = alpha[t-1] * beta[t] * trans / Z
        actual = adj + trans - Z
        assert_close(expected, actual.align(tuple(expected.inputs)), rtol=1e-4)
        print("")
        print(f"Marginal term: p(x[{t}], x[{t-1}] | Y)")
        print("Forward-backward algorithm:\n", expected.data)
        print("Differentiating forward algorithm:\n", actual.data)
        t += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward-Backward Algorithm Is Just Backprop"
    )
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-d", "--hidden-dim", default=3, type=int)
    args = parser.parse_args()

    main(args)
