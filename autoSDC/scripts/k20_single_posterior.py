#!/usr/bin/env python

import os
import sys
import json
import click
import gpflow
import numpy as np
import pandas as pd
from scipy import stats
from ruamel import yaml
from sklearn import metrics

import ternary
import matplotlib.pyplot as plt

from asdc import analyze
from asdc import emulation
from asdc import visualization

opt = gpflow.training.ScipyOptimizer()


def probability_of_improvement(mu, var, current_best=None, minimize=True):
    """ probability of improvement: default to minimization """
    dist = stats.norm(mu.flat, np.sqrt(var).flat)

    if minimize:
        poi = dist.cdf(current_best)
    else:
        poi = 1 - dist.cdf(current_best)

    return poi


def confidence_bound(mu, var, current_best, kappa=2, minimize=True):
    """ confidence bound acquisition """
    if minimize:
        bound = -(mu.flat - kappa * np.sqrt(var.flat))
    else:
        bound = mu.flat + kappa * np.sqrt(var.flat)
    return bound


def random(mu, var, current_best, minimize=True):
    """ acquisition stub for random acquisition function
    just draw acquisition function from standard gaussian distribution
    """
    N, _ = mu.shape
    return np.random.normal(0, 1, N)


def setup_acquisition(config):
    """ return a closure wrapping specific parameters of acquisition function... """
    a = config["acquisition"]
    if a["strategy"] == "cb":

        def acquisition(mu, var, current_best, minimize=True):
            return confidence_bound(
                mu,
                var,
                current_best,
                kappa=config["acquisition"]["kappa"],
                minimize=minimize,
            )

        return acquisition
    elif a["strategy"] == "random":
        return random
    elif a["strategy"] == "pi":
        return probability_of_improvement


@click.command()
@click.argument("config-file", type=click.Path())
def k20_single_objective(config_file):
    """ optimize a single-objective function emulated by a GP fit to experimental data """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_dir, _ = os.path.split(config_file)
    fig_dir = os.path.join(model_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(config)

    c = config["emulator"]
    em = emulation.ExperimentEmulator(c["datafile"], components=c["components"])

    task = config["task"]
    target = task["target"]

    evaluate_candidates = setup_acquisition(config)

    # set up a discrete grid of samples to optimize over...
    # randomize the grid because np.argmax takes the first value in memory order
    # if there are degenerate values
    domain = emulation.simplex_grid(task["domain_resolution"], buffer=task["buffer"])
    domain = domain[np.random.permutation(domain.shape[0])]

    visualization.ternary_scatter(domain, em(domain, target=target), label=target)
    plt.savefig(
        os.path.join(fig_dir, f"target_function_mean_{target}.png"), bbox_inches="tight"
    )
    plt.clf()

    # find max
    # s = emulation.simplex_grid(200, buffer=0.05)
    response = em(domain, target=target, sample_posterior=True)
    max_value = response.max()
    min_value = response.min()
    print(max_value)

    visualization.ternary_scatter(domain, response, label=target)
    plt.savefig(
        os.path.join(fig_dir, f"target_function_{target}.png"), bbox_inches="tight"
    )
    plt.clf()

    # initialize with the corners of the simplex subdomain...
    queries = []
    _s = domain.argmax(axis=0)
    s = domain[_s]
    v = response[_s]

    mae, r2, ev, best_value = [], [], [], []
    for query_idx in range(task["budget"]):

        # draw a picture
        visualization.ternary_scatter(s, v)
        plt.savefig(
            os.path.join(fig_dir, f"measured_{target}_{len(queries):02d}.png"),
            bbox_inches="tight",
        )
        plt.clf()

        # fit the surrogate model
        m = emulation.model_ternary(s, v[:, None])
        opt.minimize(m)
        mu, var = m.predict_y(domain[:, :-1])

        # assess regret...
        if task["minimize"]:
            current_best = v.min()
            print(f"query {query_idx}: {min_value - current_best}")
        else:
            current_best = v.max()
            print(f"query {query_idx}: {max_value - current_best}")

        # evaluate predictive accuracy
        best_value.append(current_best)
        mae.append(np.mean(np.abs(response - mu)))
        r2.append(metrics.r2_score(response.flat, mu.flat))
        ev.append(metrics.explained_variance_score(response.flat, mu.flat))
        print("MAE", mae[-1])
        print("R2", r2[-1])
        print("EV", ev[-1])
        plt.scatter(response.flat, mu.flat)
        plt.plot(
            (response.min(), response.max()),
            (response.min(), response.max()),
            linestyle="--",
            color="k",
        )
        plt.savefig(
            os.path.join(fig_dir, f"parity_{target}_{len(queries):02d}.png"),
            bbox_inches="tight",
        )
        plt.clf()

        # draw the extrapolations
        visualization.ternary_scatter(domain, mu.flat, label=target)
        plt.savefig(
            os.path.join(fig_dir, f"surrogate_{target}_{len(queries):02d}.png"),
            bbox_inches="tight",
        )
        plt.clf()

        acquisition = evaluate_candidates(
            mu, var, current_best, minimize=task["minimize"]
        )
        acquisition[queries] = -np.inf

        visualization.ternary_scatter(domain, acquisition, label="acquisition")
        plt.savefig(
            os.path.join(fig_dir, f"acquisition_{target}_{len(queries):02d}.png"),
            bbox_inches="tight",
        )
        plt.clf()

        # update the dataset
        # index into pre-generated posterior function sample...
        query_idx = np.argmax(acquisition)
        queries.append(query_idx)
        query = domain[queries[-1]][None, :]
        s = np.vstack((s, query))
        v = np.hstack((v, response[query_idx]))

    # draw a picture
    visualization.ternary_scatter(s, v)
    plt.savefig(
        os.path.join(fig_dir, f"measured_{target}_{len(queries)}.png"),
        bbox_inches="tight",
    )
    plt.clf()

    budget = (3 + np.arange(task["budget"])).tolist()
    with open(os.path.join(model_dir, "trace.json"), "w") as f:
        json.dump(
            {
                "budget": budget,
                "mae": mae,
                "r2": r2,
                "ev": ev,
                "best_value": best_value,
            },
            f,
        )

    return


if __name__ == "__main__":
    k20_single_objective()
