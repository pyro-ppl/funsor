# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import funsor
import funsor.ops as ops
from funsor import Funsor, Tensor
from funsor.adjoint import AdjointTape
from funsor.domains import Bint
from funsor.ops.builtin import UNITS, AssociativeOp
from funsor.testing import assert_close, random_tensor


def forward_algorithm(
    sum_op: AssociativeOp,
    prod_op: AssociativeOp,
    factors: List[Funsor],
    step: Dict[str, str],
) -> Tuple[Funsor, List[Funsor]]:
    """
    Calculate log marginal probability using the forward algorithm:
    log_Z = log p(y[0:T])

    Transition and emission probabilities are given by init and trans:
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
        alpha = prod_op(alpha(**curr_to_drop), trans(**prev_to_drop)).reduce(
            sum_op, reduce_vars
        )
        alphas.append(alpha)
    else:
        Z = alpha(**curr_to_drop).reduce(sum_op, reduce_vars)
    # log_Z = Z.log()
    return Z


def forward_backward_algorithm(
    sum_op: AssociativeOp,
    prod_op: AssociativeOp,
    factors: List[Funsor],
    step: Dict[str, str],
) -> List[Tensor]:
    """
    Calculate marginal probabilities:
    p(x[t], x[t-1] | Y)
    beta[t] = p(y[t+1:T]|x[t])
    """
    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    reduce_vars = frozenset(drop)

    # Backward algorithm
    # beta[T] = 1
    inputs = factors[0](**{"x_curr": "x_prev"}).inputs.copy()
    data = funsor.ops.new_zeros(funsor.tensor.get_default_prototype(), ()).expand(
        tuple(v.size for v in inputs.values())
    )
    data = data + UNITS[prod_op]
    beta = Tensor(data, inputs, factors[-1].dtype)
    betas = [beta]
    # inductive steps
    # beta[t] = trans[t+1] @ beta[t+1]
    for trans in factors[:0:-1]:
        beta = prod_op(trans(**curr_to_drop), beta(**prev_to_drop)).reduce(
            sum_op, reduce_vars
        )
        betas.append(beta)
    else:
        init = factors[0]
        Z = prod_op(init(**curr_to_drop), beta(**prev_to_drop)).reduce(
            sum_op, reduce_vars
        )
    betas.reverse()

    # Forward algorithm
    # p(y[0], x[0])
    alpha = factors[0]
    alphas = [alpha]
    # p(x[0] | Y)
    marginal = prod_op(factors[0], betas[0](**{"x_prev": "x_curr"})) / Z
    marginals = [marginal]
    # inductive steps
    for trans, beta in zip(factors[1:], betas[1:]):
        # p(y[0:t], x[t-1], x[t])
        new_term = prod_op(alpha(**{"x_curr": "x_prev"}), trans)
        # p(y[0:t], x[t])
        alpha = new_term.reduce(sum_op, frozenset({"x_prev"}))
        alphas.append(alpha)
        # p(x[t-1], x[t] | Y)
        marginal = prod_op(new_term, beta(**{"x_prev": "x_curr"})) / Z
        marginals.append(marginal)

    return marginals


def main(args):
    """
    Compute marginal probabilities p(x[t], x[t-1] | Y) for an HMM:

    x[0] -> ... -> x[t-1] -> x[t] -> ... -> x[T]
     |              |         |             |
     v              v         v             v
    y[0]           y[t-1]    y[t]           y[T]

    alpha[t] = alpha[t-1] @ p(y[t], x[t] | x[t-1])
    beta[t-1] = p(y[t], x[t] | x[t-1]) @ beta[t]

    dlog_Z / dlog_p(y[t], x[t] | x[t-1]) =
    alpha[t-1] * beta[t] * p(y[t], x[t] | x[t-1]) / Z =
    p(x[t], x[t-1] | Y)

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
    funsor.set_backend("torch")

    # transition and emission probabilities
    init = random_tensor(OrderedDict([("x_curr", Bint[args.hidden_dim])]))
    log_factors = [init]
    for t in range(args.time_steps - 1):
        log_factors.append(
            random_tensor(
                OrderedDict(
                    [
                        ("x_prev", Bint[args.hidden_dim]),
                        ("x_curr", Bint[args.hidden_dim]),
                    ]
                )
            )
        )
    factors = [f.exp() for f in log_factors]

    # Compute marginal probabilities using the forward-backward algorithm
    marginals = forward_backward_algorithm(
        ops.add, ops.mul, factors, {"x_prev": "x_curr"}
    )
    # Compute marginal probabilities using backpropagation
    with AdjointTape() as tape:
        log_Z = forward_algorithm(ops.logaddexp, ops.add, log_factors, {"x_prev": "x_curr"})
    result = tape.adjoint(ops.logaddexp, ops.add, log_Z, log_factors)
    adjoint_marginals = list(result.values())

    print("Smoothed term")
    print("Forward-backward algorithm")
    print("Differentiating backward algorithm")
    t = 0
    for v1, v2 in zip(marginals, adjoint_marginals):
        breakpoint()
        v2 = (v2 - log_Z).exp()
        assert_close(v1, v2)
        print("")
        print(f"gamma[{t}]")
        print(v1.data)
        print(v2.data)
        t += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward-Backward Algorithm Is Just Backprop"
    )
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-d", "--hidden-dim", default=3, type=int)
    args = parser.parse_args()

    main(args)
