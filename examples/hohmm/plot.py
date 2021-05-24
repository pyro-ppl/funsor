
import seaborn as sns

import pyro
from pyro.ops.indexing import vindex


def main():
    pyro.get_param_store().load("params_10.pt")
    probs_xlag = pyro.param("probs_xlag")
    mix_lambda = pyro.param("mix_lambda")
    mu = pyro.param("AutoDelta.mu")
    sigma = pyro.param("sigma")
    probs_x = 0.
    for i in range(10):
        probs_x = probs_x + mix_lambda[i] * probs_xlag[i].reshape((3,) + (1,) * i + (3,))
    samples = []
    for i in range(1000):
        preds = []
        # using the last 10 data points
        x_prevs, x_curr = [2, 2, 2, 2, 2, 2, 2, 1, 2], 1
        for t in range(3):
            probs_x_t = vindex(probs_x, tuple(x_prevs) + (x_curr,))
            x_prevs = x_prevs[1:] + [x_curr]
            x_curr = pyro.sample("x_{}".format(t), dist.Categorical(probs_x_t))
            preds.append(pyro.sample("y_{}".format(t), dist.Normal(mu[x_curr], sigma[x_curr])))
        samples.append(torch.stack(preds))
    samples = torch.stack(samples)
    # TODO: plot the samples

