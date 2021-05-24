import argparse
from functools import partial
import logging
import sys

import pandas as pd
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.infer import autoguide, SVI, TraceEnum_ELBO
from pyro.ops.indexing import Vindex, vindex
import pyro.poutine as poutine


logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


def hohmm(data, history=1):
    x_dim = 3

    # Use the more parsimonious "Raftery" parameterization of
    # the tensor of transition probabilities. See reference:
    # Raftery, A. E. A model for high-order markov chains.
    # Journal of the Royal Statistical Society. 1985.
    probs_xlag = pyro.param("probs_xlag", torch.ones(history, x_dim, x_dim) / x_dim, constraint=constraints.simplex)
    if history > 1:
        mix_lambda = pyro.param("mix_lambda", torch.ones(history) / history, constraint=constraints.simplex)
    else:
        mix_lambda = [1.]
    # we use broadcasting to combine two tensors of shape (hidden_dim, hidden_dim) and
    # (hidden_dim, 1, hidden_dim) to obtain a tensor of shape (hidden_dim, hidden_dim, hidden_dim)
    probs_x = 0.
    for i in range(history):
        probs_x = probs_x + mix_lambda[i] * probs_xlag[i].reshape((x_dim,) + (1,) * i + (x_dim,))

    mu = pyro.sample("mu", dist.Normal(torch.tensor([-0.5, 0., 0.5]), torch.tensor([1/6, 1e-5, 1/6])).to_event(1))
    # TODO: use Gamma(...) for precision
    sigma = pyro.param("sigma", torch.tensor(0.05).repeat(3), constraint=constraints.interval(0., 0.1))

    # initial states are 1, which has corresponding mean is 0
    x_prevs, x_curr = [torch.tensor(1) for i in range(history - 1)], torch.tensor(1)
    for t in pyro.markov(range(len(data)), history=history):
        probs_x_t = vindex(probs_x, tuple(x_prevs) + (x_curr,))
        x_prevs = x_prevs[1:] + [x_curr]
        x_curr = pyro.sample("x_{}".format(t), dist.Categorical(probs_x_t), infer={"enumerate": "parallel"})
        pyro.sample("y_{}".format(t), dist.Normal(mu[x_curr], sigma[x_curr]), obs=data[t])


models = {str(i): partial(hohmm, history=i)
          for i in range(1, 11)}


def main(args):
    logging.info('Loading data')
    data = torch.from_numpy(pd.read_csv("aCGH.csv").fillna(0.).values).float()

    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} with history={}'.format(
        "hohmm", args.model))

    if args.truncate:
        data = data[:args.truncate]
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    guide = autoguide.AutoDelta(pyro.poutine.block(model, expose=["mu"]))

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    if args.print_shapes:
        first_available_dim = -1
        guide_trace = poutine.trace(guide).get_trace(data)
        model_trace = poutine.trace(
            poutine.replay(poutine.enum(model, first_available_dim), guide_trace)).get_trace(data)
        logging.info(model_trace.format_shapes())

    optim = pyro.optim.Adam({'lr': args.learning_rate})
    elbo = TraceEnum_ELBO(max_plate_nesting=0)
    svi = SVI(model, guide, optim, elbo)

    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(data)
        logging.info("probs_xlag = {}".format(pyro.param("probs_xlag").detach().clone()))
        if int(args.model) > 1:
            logging.info("mix_lambda = {}".format(pyro.param("mix_lambda").detach().clone()))
        logging.info("sigma = {}".format(pyro.param("sigma").detach().clone()))
        logging.info("mu = {}".format(pyro.param("AutoDelta.mu").detach().clone()))
        logging.info('{: >5d}\t{}'.format(step, loss))

    pyro.get_param_store().save("params_{}.pt".format(args.model))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.6.0')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning aCGH")
    parser.add_argument("-m", "--model", default="2", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)
