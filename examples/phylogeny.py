# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import math
from typing import List, Tuple

from torch.distributions import Poisson

import funsor
import funsor.ops as ops
from funsor import Bint, Real, Variable, function, to_funsor
from funsor.util import fv

num_sites = 10
birth_rate = 0.1
death_rate = 0.1
sample_rate = 0.01


def ids_to_name(ids):
    return ",".join(map(str, sorted(ids)))


def p_init(x):
    raise NotImplementedError("TODO")


def p_transition(dt, x0, x1):
    raise NotImplementedError("TODO")


def p_coalesce(dt):
    raise NotImplementedError("TODO")


def markov_reduce(op):
    """
    Decorator to reduce all local variables of a function.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            result = fn(*args, **kwargs)
            local = fv(result) - fv(args) - fv(kwargs)
            result = result.reduce(op, local)
            return result
        return decorated
    return decorator


# TODO refactor @function to be a dynamic Op factory
@funsor.function
def Partitions(ids: Bint["n"]) -> List[Tuple[FIXME, FIXME]]:
    ids = frozenset(ids)
    for k in range(1, len(ids)):
        for lhs in itertools.combinations(ids, k):
            lhs = tuple(sorted(lhs))
            rhs = tuple(sorted(ids.difference(lhs)))
            yield lhs, rhs


@markov_reduce(ops.logaddexp)
def coalescent(ts, xs):
    # Initialize the tree.
    if len(ts) == 1:
        x = xs[0]
    else:
        name = ids_to_name(range(len(ts)))
        x = Variable(f"x{name}", Bint[4, num_sites])["site"]
    joint = p_init(x).reduce(ops.add, "site")

    joint += _coalescent(ts, xs)
    return joint


@funsor.util.weak_memoize
@markov_reduce(ops.logaddexp)
def _coalescent(ts, xs, ids):
    name = ids_to_name(ids)

    @funsor.root_function
    def loop_body(split: Bint[2 ** (len(ids) - 1)]):
        lhs, rhs = Partitions(ids)[split]

        lhs = tuple(sorted(lhs))
        rhs = tuple(sorted(rhs))

        # Assume the tree splits into two subtrees:
        #     root
        #    /    \
        #  lhs    rhs   } these are trees
        joint = to_funsor(0)
        if lhs and rhs:
            lhs_name = ids_to_name(lhs)
            rhs_name = ids_to_name(rhs)
            t_root = Variable(f"t{name}", Real)
            x_root = Variable(f"x{name}", Bint[4, num_sites])["site"]
            if len(lhs) == 1:
                t_lhs = ts[lhs[0]]
                x_lhs = xs[lhs[0]]
            else:
                t_lhs = Variable(f"t{lhs_name}", Real)
                x_lhs = Variable(f"x{lhs_name}", Bint[4, num_sites])["site"]
            if len(rhs) == 1:
                t_rhs = ts[rhs[0]]
                x_rhs = xs[rhs[0]]
            else:
                t_rhs = Variable(f"t{rhs_name}", Real)
                x_rhs = Variable(f"x{rhs_name}", Bint[4, num_sites])["site"]

            # Add current step.
            joint += p_coalesce(t_root - ops.max(t_lhs, t_rhs))
            joint += p_transition(t_lhs - t_root, x_root, x_lhs).reduce(ops.add, "site")
            joint += p_transition(t_rhs - t_root, x_root, x_rhs).reduce(ops.add, "site")

            # Recurse
            joint += _coalescent(ts, xs, lhs)
            joint += _coalescent(ts, xs, rhs)
        return joint

    # Sum over all combinations.
    joint = loop_body(split=f"split{name}")
    return joint


@markov_reduce(ops.logaddexp)
def birth_death(ts, xs, ids):
    name = ids_to_name(ids)
    t = Variable(f"t{name}")
    x = Variable(f"x{name}", Bint[4, num_sites])["site"]
    joint = p_init(x).reduce(ops.add, "site")
    joint += _birth_death(ts, xs, ids, t, x)
    return joint


@markov_reduce(ops.logaddexp)
def _birth_death(ts, xs, ids, t_prev, x_prev):
    joint = to_funsor(0)
    name = ids_to_name(ids)

    # Sample time to next event.
    dt = Variable(f"dt_{name}", Real)
    joint += Poisson(birth_rate + death_rate + sample_rate).log_prob(dt)
    event_type = Variable(f"event_type_{name}", Bint[3])
    if event_type == 0:  # Birth.
        t = t_prev + dt
        x = Variable(f"x{name}", Bint[4, num_sites])["site"]
        joint += p_transition(dt, x_prev, x).reduce(ops.add, "site")

        @function
        def loop_body(split: Bint[2 ** (len(ids) - 1)]) -> Real:
            lhs, rhs = Partitions(ids)[split]
            joint = to_funsor(0)
            if lhs:
                joint += _birth_death(ts, xs, lhs, t, x)
            if rhs:
                joint += _birth_death(ts, xs, rhs, t, x)
            return joint

        joint += loop_body(split=f"split{name}")
    elif event_type == 1:  # Death.
        pass  # implicit
    elif event_type == 2:  # Sampling and death.
        if len(ids) != 1:
            return to_funsor(-math.inf)
        dt = t - t_prev
        t = ts[ids[0]]
        x = xs[ids[0]]
        joint += p_transition(dt, x_prev, x)

    return joint
