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
    init: Funsor,
    trans: Funsor,
    time: str,
    step: Dict[str, str],
) -> Tuple[Funsor, List[Funsor]]:
    """
    Calculate forward probabilities defined as:
    alpha[t] = p(y[1:t], x[t])

    Transition and emission probabilities are given by init and trans:
    init = p(y[0], x[0])
    trans[t] = p(y[t], x[t]|x[t-1])

    Forward probabilities are computed inductively:
    alpha[0] = init
    alpha[t+1] = alpha[t] @ trans[t+1]
    """
    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    reduce_vars = frozenset(drop)

    factors = [trans(**{time: t}) for t in range(trans.inputs[time].size)]
    factors.reverse()
    # base case
    base = init
    alphas = [base]
    # inductive steps
    while factors:
        y = factors.pop()(**prev_to_drop)
        base = prod_op(base(**curr_to_drop), y).reduce(sum_op, reduce_vars)
        alphas.append(base)
    else:
        base = base(**curr_to_drop).reduce(sum_op, reduce_vars)
    return base, alphas


def backward_algorithm(
    sum_op: AssociativeOp,
    prod_op: AssociativeOp,
    init: Funsor,
    trans: Funsor,
    time: str,
    step: Dict[str, str],
) -> Tuple[Tensor, List[Tensor]]:
    """
    Calculate backward probabilities defined as:
    beta[t] = p(y[t+1:T]|x[t])

    Transition and emission probabilities are given by init and trans:
    init = p(y[0], x[0])
    trans[t] = p(y[t], x[t]|x[t-1])

    Backward probabilities are computed inductively:
    beta[T] = 1
    beta[t] = trans[t+1] @ beta[t+1]
    """
    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    reduce_vars = frozenset(drop)

    factors = [trans(**{time: t}) for t in range(trans.inputs[time].size)]
    # base case
    inputs = OrderedDict((k, v) for k, v in trans.inputs.items() if k in step.keys())
    data = funsor.ops.new_zeros(funsor.tensor.get_default_prototype(), ()).expand(
        tuple(v.size for v in inputs.values())
    )
    data = data + UNITS[prod_op]
    base = Tensor(data, inputs, trans.dtype)
    betas = [base]
    # inductive steps
    while factors:
        x = factors.pop()(**curr_to_drop)
        base = prod_op(x, base(**prev_to_drop)).reduce(sum_op, reduce_vars)
        betas.append(base)
    else:
        base = prod_op(init(**curr_to_drop), base(**prev_to_drop)).reduce(
            sum_op, reduce_vars
        )
    betas.reverse()
    return base, betas


def main(args):
    """
    Compute smoothed values (gamma[t] = p(x[t]|y[1:T])) for an HMM:

    x[0] -> ... -> x[t-1] -> x[t] -> ... -> x[T]
     |              |         |             |
     v              v         v             v
    y[0]           y[t-1]    y[t]           y[T]

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
    sum_op = ops.logaddexp
    prod_op = ops.add

    # transition and emission probabilities
    time = "time"
    init = random_tensor(OrderedDict([("x_curr", Bint[args.hidden_dim])]))
    trans = random_tensor(
        OrderedDict(
            [
                (time, Bint[args.time_steps]),
                ("x_prev", Bint[args.hidden_dim]),
                ("x_curr", Bint[args.hidden_dim]),
            ]
        )
    )

    # Compute smoothed values by using the forward-backward algorithm
    # gamma[t] = p(x[t]|y[1:T])
    # gamma[t] = alpha[t] * beta[t]
    Z_forward, alphas = forward_algorithm(
        sum_op, prod_op, init, trans, time, {"x_prev": "x_curr"}
    )
    Z_backward, betas = backward_algorithm(
        sum_op, prod_op, init, trans, time, {"x_prev": "x_curr"}
    )
    assert_close(Z_forward, Z_backward)
    gammas = []
    for alpha, beta in zip(alphas, betas):
        gamma = prod_op(alpha(**{"x_curr": "x_prev"}), beta)
        gammas.append(gamma)

    # forward-backward algorithm can be obtained by differentiating the backward algorithm
    with AdjointTape() as tape:
        Z_backward, betas = backward_algorithm(
            sum_op, prod_op, init, trans, time, {"x_prev": "x_curr"}
        )
    result = tape.adjoint(sum_op, prod_op, Z_backward, betas)
    adjoint_gammas = list(result.values())

    print("Smoothed term")
    print("Forward-backward algorithm")
    print("Differentiating backward algorithm")
    t = 0
    for v1, v2 in zip(gammas, adjoint_gammas):
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
