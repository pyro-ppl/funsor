# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple
from functools import reduce

import funsor
import funsor.ops as ops
from funsor import Funsor, Tensor, Cat, Slice, Variable
from funsor.cnf import Contraction
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
    alpha[t] = p(y[0:t], x[t])

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
    # reduce_vars = frozenset(drop)
    reduce_vars = frozenset(Variable(v, trans.inputs[k])
                     for k, v in curr_to_drop.items())

    duration = trans.inputs[time].size
    while duration > 1:
        even_duration = duration // 2 * 2
        x = trans(**{time: Slice(time, 0, even_duration, 2, duration)}, **curr_to_drop)
        y = trans(**{time: Slice(time, 1, even_duration, 2, duration)}, **prev_to_drop)
        contracted = Contraction(sum_op, prod_op, reduce_vars, x, y)

        if duration > even_duration:
            extra = trans(**{time: Slice(time, duration - 1, duration)})
            contracted = Cat(time, (contracted, extra))
        trans = contracted
        duration = (duration + 1) // 2
    else:
        Z = prod_op(init(**{"x_curr": "x_prev"}), trans(**{time: 0})).reduce(sum_op, frozenset({"x_prev", "x_curr"}))
    return Z

    #  # base case
    #  alpha = init
    #  alphas = [alpha]
    #  ys = []
    #  # inductive steps
    #  for t in range(trans.inputs[time].size):
    #      y = trans(**{time: t})
    #      alpha = prod_op(alpha(**curr_to_drop), y(**prev_to_drop)).reduce(sum_op, reduce_vars)
    #      alphas.append(alpha)
    #      ys.append(y)
    #  else:
    #      Z = alpha(**curr_to_drop).reduce(sum_op, reduce_vars)
    #  return Z, ys


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

    # base case
    inputs = OrderedDict((k, v) for k, v in trans.inputs.items() if k in step.keys())
    data = funsor.ops.new_zeros(funsor.tensor.get_default_prototype(), ()).expand(
        tuple(v.size for v in inputs.values())
    )
    data = data + UNITS[prod_op]
    beta = Tensor(data, inputs, trans.dtype)
    betas = [beta]
    # inductive steps
    for t in reversed(range(trans.inputs[time].size)):
        x = trans(**{time: t}, **curr_to_drop)
        beta = prod_op(x, beta(**prev_to_drop)).reduce(sum_op, reduce_vars)
        betas.append(beta)
    else:
        Z = prod_op(init(**curr_to_drop), beta(**prev_to_drop)).reduce(
            sum_op, reduce_vars
        )
    betas.reverse()

    # base case
    alpha = init
    alphas = [alpha]
    marginals = []
    # inductive steps
    for t in range(trans.inputs[time].size):
        y = trans(**{time: t}, **prev_to_drop)
        alpha = prod_op(alpha(**curr_to_drop), y).reduce(sum_op, reduce_vars)
        alphas.append(alpha)
        marginal = reduce(prod_op, [alphas[t](**{"x_curr": "x_prev"}), trans(**{time: t}), betas[t+1](**{"x_prev": "x_curr"})])
        marginals.append(marginal)
    else:
        Z = alpha(**curr_to_drop).reduce(sum_op, reduce_vars)

    return Z, betas, marginals


def main(args):
    """
    Compute smoothed values (gamma[t] = p(x[t]|y[0:T])) for an HMM:

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
    sum_op = ops.add
    prod_op = ops.mul

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
    # gamma[t] = p(x[t]|y[0:T])
    # gamma[t] = alpha[t] * beta[t]
    with AdjointTape() as forward_tape:
        Z_forward = forward_algorithm(
            sum_op, prod_op, init, trans, time, {"x_prev": "x_curr"}
        )
    with AdjointTape() as backward_tape:
        Z_backward, betas, marginals = backward_algorithm(
            sum_op, prod_op, init, trans, time, {"x_prev": "x_curr"}
        )
    assert_close(Z_forward, Z_backward)
    #  gammas = []
    #  for alpha, beta in zip(alphas, betas):
    #      gamma = prod_op(alpha, beta(**{"x_prev": "x_curr"}))
    #      # gamma = prod_op(alpha(**{"x_curr": "x_prev"}), beta)
    #      gammas.append(gamma)

    # forward-backward algorithm can be obtained by differentiating the backward algorithm
    #  bresult = backward_tape.adjoint(sum_op, prod_op, Z_backward, betas)
    #  fresult = forward_tape.adjoint(sum_op, prod_op, Z_forward, alphas)
    result = forward_tape.adjoint(sum_op, prod_op, Z_forward, [trans])
    #  adjoint_gammas = list(fresult.values())
    adjoint_marginals = list(result.values())
    breakpoint()

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
