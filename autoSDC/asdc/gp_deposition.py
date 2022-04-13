""" GP deposition -- deposit duplicate samples, performing a corrosion test on the second """
import os
import sys
import json
import time
import click
import asyncio
import dataset
import functools
import numpy as np
import pandas as pd
from ruamel import yaml
from aioconsole import ainput

from typing import Any, List, Dict, Optional, Tuple

import matplotlib.pyplot as plt

import gpflow
import gpflowopt
from gpflowopt import acquisition
from scipy import stats
from scipy import spatial
from scipy import integrate
from datetime import datetime

sys.path.append(".")

import cycvolt
from asdc import _slack
from asdc import slackbot
from asdc import analyze
from asdc import emulation
from asdc import visualization
from asdc import characterization

import enum

Action = enum.Enum("Action", ["QUERY", "REPEAT", "PHOTONS", "CORRODE"])

BOT_TOKEN = open("slack_bot_token.txt", "r").read().strip()
SDC_TOKEN = open("slacktoken.txt", "r").read().strip()


def deposition_instructions(query, experiment_id=0):
    """ TODO: do something about deposition duration.... """
    # query = pd.Series({'Ni_fraction': guess[0], 'metal_fraction': guess[1], 'potential': guess[2]})

    total_flow = 0.15
    relative_rates = {
        "KCl": (1 - query["metal_fraction"]),
        "NiCl2": query["metal_fraction"] * query["Ni_fraction"],
        "ZnCl2": query["metal_fraction"] * (1 - query["Ni_fraction"]),
    }
    rates = {key: value * total_flow for key, value in relative_rates.items()}

    instructions = [
        {"intent": "deposition", "experiment_id": experiment_id},
        {"op": "set_flow", "rates": rates, "hold_time": 180},
        {
            "op": "potentiostatic",
            "potential": query["potential"],
            "duration": 600,
            "current_range": "2MA",
        },
        {"op": "post_flush", "rates": {"H2O": 1.0}, "duration": 120},
    ]
    return instructions


def characterize_instructions(experiment_id=0):
    return [
        {"intent": "characterize", "experiment_id": experiment_id},
        {"op": "photons"},
    ]


def corrosion_instructions(experiment_id=0):
    instructions = [
        {"intent": "corrosion", "experiment_id": experiment_id},
        {"op": "set_flow", "rates": {"NaCl": 0.1}, "hold_time": 120},
        {
            "op": "lpr",
            "initial_potential": -0.03,
            "final_potential": 0.03,
            "step_size": 0.0005,
            "step_time": 1.0,
            "current_range": "2MA",
        },
        # {
        #     "op": "lsv",
        #     "initial_potential": -1.0,
        #     "final_potential": 0.5,
        #     "scan_rate": 0.075,
        #     "current_range": "2MA"
        # },
        {"op": "post_flush", "rates": {"H2O": 0.5}, "duration": 90},
    ]
    return instructions


def exp_id(db):
    """we're running two depositions followed by a corrosion experiment
    return 0 if it's time for the first deposition
           1 if it's time for  the second deposition
           2 if it's time for corrosion
    """
    deps = db["experiment"].count(intent="deposition")
    cors = db["experiment"].count(intent="corrosion")

    if cors == 0:
        phase = deps
    else:
        phase = deps % cors

    if phase in (0, 1):
        intent = "deposition"
    else:
        intent = "corrosion"
    if phase == 0:
        fit_gp = True
    else:
        fit_gp = False

    return intent, fit_gp


def select_action(db, run_replicates=True, threshold=0.9):
    """run two depositions, followed by a corrosion experiment if the deposits are acceptable."""
    prev_id = db["experiment"].count()

    prev = db["experiment"].find_one(id=prev_id)

    if prev["intent"] == "corrosion":
        return Action.QUERY

    elif prev["intent"] == "deposition":
        n_repeats = db["experiment"].count(experiment_id=prev["experiment_id"])

        if n_repeats == 1:
            # logic to skip replicate based on quality goes here...
            return Action.REPEAT

        elif n_repeats == 2:
            # if coverage is good enough, run a corrosion measurement.
            session = pd.DataFrame(
                db["experiment"].find(experiment_id=prev["experiment_id"])
            )
            min_coverage = session["coverage"].min()

            if min_coverage >= threshold:
                target = pd.DataFrame(
                    db["experiment"].find(experiment_id=prev["experiment_id"])
                )
                target = target[~(target["has_bubble"] == True)]

                if target.shape[0] == 0:
                    print("no replicates without bubbles...")
                    return Action.QUERY
                else:
                    target = target.iloc[0]
                    print(f"good coverage ({min_coverage})")
                    print("target", target["id"])
                    pos = {"x": target["x_combi"], "y": target["y_combi"]}
                    return Action.CORRODE
            else:
                print(f"poor coverage ({min_coverage})")
                return Action.QUERY


def select_action_single(db, run_replicates=True, threshold=0.9):
    """run single depositions, followed by a corrosion experiment if the deposits are acceptable."""
    prev_id = db["experiment"].count()

    prev = db["experiment"].find_one(id=prev_id)

    if prev["intent"] == "corrosion":
        return Action.QUERY

    elif prev["intent"] == "deposition":
        expt = db["experiment"].find_one(experiment_id=prev["experiment_id"])

        if expt["image_name"] is None:
            return Action.PHOTONS

        return Action.CORRODE


def load_cv(row, data_dir="data", segment=2, half=True, log=True):
    """ load CV data and process it... """
    cv = pd.read_csv(os.path.join(data_dir, row["datafile"]), index_col=0)

    sel = cv["segment"] == segment
    I = cv["current"][sel].values
    V = cv["potential"][sel].values
    t = cv["elapsed_time"][sel].values

    if half:
        # grab the length of the polarization curve
        n = I.size // 2
        I = I[:n]
        V = V[:n]

    if log:
        I = cycvolt.analyze.log_abs_current(I)

    return V, I, t - t[0]


def deposition_flow_rate(ins):
    i = json.loads(ins)
    try:
        return i[0]["rates"]["CuSO4"]
    except KeyError:
        return None


def deposition_potential(df):
    p = []
    for idx, row in df.iterrows():

        if row["intent"] == "deposition":
            instructions = json.loads(row["instructions"])
            for instruction in json.loads(row["instructions"]):
                if instruction.get("op") == "potentiostatic":
                    p.append(instruction.get("potential"))
        elif row["intent"] == "corrosion":
            p.append(None)
    return p


def load_experiment_files(csv_files, dir="."):
    dir, _ = os.path.split(dir)
    file = os.path.join(dir, csv_file)
    if os.path.isfile(file):
        experiments = pd.concat(
            (pd.read_csv(file, index_col=0) for csv_file in csv_files),
            ignore_index=True,
        )
    else:
        experiments = []
    return experiments


def load_experiment_json(experiment_files, dir="."):
    """ an experiment file contains a json list of experiment definitions """
    dir, _ = os.path.split(dir)

    experiments = None
    for experiment_file in experiment_files:
        p = os.path.join(dir, experiment_file)
        if os.path.isfile(p):
            with open(p, "r") as f:
                if experiments is None:
                    experiments = json.load(f)
                else:
                    experiments.append(json.load(f))
        else:
            experiments = []

    return experiments


def confidence_bound(model, candidates, sign=1, cb_beta=0.25):
    # set per-model confidence bound beta
    # default to lower confidence bound
    t = model.X.shape[0]
    cb_weight = cb_beta * np.log(2 * t + 1)

    mean, var = model.predict_y(candidates)
    criterion = (sign * mean) - cb_weight * np.sqrt(var)
    return criterion


def composition_loss_confidence_bound(model, candidates, target, sign=1, cb_beta=0.25):
    # set per-model confidence bound beta
    # default to lower confidence bound
    t = model.X.shape[0]
    cb_weight = cb_beta * np.log(2 * t + 1)

    mean, var = model.predict_y(candidates)
    composition_loss = np.abs(mean - target)
    criterion = composition_loss - cb_weight * np.sqrt(var)

    return criterion


def classification_criterion(model, candidates):
    """ compute the classification criterion from 10.1007/s11263-009-0268-3 """
    loc, scale = model.predict_f(candidates)
    criterion = np.abs(loc) / np.sqrt(scale + 0.001)
    return criterion


def plot_map(vals, X, guess, extent, figpath):
    plt.figure(figsize=(4, 4))
    plt.imshow(vals, cmap="Blues", extent=extent, origin="lower")
    plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], color="k")
    plt.scatter(guess[0], guess[1], color="r")

    if "coverage" in figpath:
        plt.contour(vals, levels=[0.5], extent=extent, colors="k", linestyles="--")

    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.xlabel("flow rate")
    plt.ylabel("potential")
    plt.tight_layout()
    plt.savefig(figpath, bbox_inches="tight")
    plt.clf()


def filter_experiments(instructions, num_previous):
    """ filter out instructions files -- count only operations that add rows to the db """
    expt_count = 0

    if num_previous == 0:
        return instructions

    for idx, expt in enumerate(instructions):
        intent = expt[0].get("intent")
        if intent in ("deposition", "corrosion"):
            if expt_count == num_previous:
                break
            expt_count += 1
            print(expt_count, intent)

    return instructions[idx:]


class Controller(slackbot.SlackBot):
    """ autonomous scanning droplet cell client """

    command = slackbot.CommandRegistry()

    def __init__(self, config=None, verbose=False, logfile=None, token=BOT_TOKEN):
        super().__init__(name="ctl", token=token)
        self.command.update(super().command)
        self.msg_id = 0
        # self.update_event = asyncio.Event(loop=self.loop)

        self.verbose = verbose
        self.logfile = logfile

        self.confirm = config.get("confirm", True)
        self.notify = config.get("notify_slack", True)
        self.data_dir = config.get("data_dir", os.getcwd())
        self.figure_dir = config.get("figure_dir", os.getcwd())
        self.domain_file = config.get("domain_file")
        self.coverage_threshold = config.get("coverage_threshold", 0.9)

        self.repeat_depositions = config.get("repeat_depositions", False)

        self.db_file = os.path.join(self.data_dir, config.get("db_file", "test.db"))
        self.db = dataset.connect(f"sqlite:///{self.db_file}")
        self.experiment_table = self.db["experiment"]

        self.targets = pd.read_csv(config["target_file"], index_col=0)
        instructions = load_experiment_json(
            config["experiment_file"], dir=self.data_dir
        )

        # remove experiments if there are records in the database
        num_previous = self.db["experiment"].count()
        self.experiments = filter_experiments(instructions, num_previous)

        # gpflowopt minimizes objectives...
        # UCB switches to maximizing objectives...
        # classification criterion: minimization
        # confidence bound using LCB variant
        # swap signs for things we want to maximize (just polarization_resistance...)
        self.objectives = ("Ni_loss", "Ni_variance", "polarization_resistance")
        self.objective_alphas = [3, 2, 1]
        # self.objective_alphas = [1, 1, 1]
        self.sgn = np.array([1, 1, -1])

        # set up the optimization domain
        with open(os.path.join(self.data_dir, os.pardir, self.domain_file), "r") as f:
            domain_data = json.load(f)

        dmn = domain_data["domain"]["x1"]
        # self.levels = [
        #     np.array([0.030, 0.050, 0.10, 0.30]),
        #     np.linspace(dmn['min'], dmn['max'], 50)
        # ]
        self.levels = [
            np.linspace(0.0, 1.0, 100),  # Ni fraction in solution
            np.linspace(0.1, 1.0, 100),  # metal fraction in solution
            np.linspace(-1.5, -1.3, 50),  # deposition potential
        ]
        self.ndim = [len(l) for l in self.levels][::-1]
        self.extent = [
            np.min(self.levels[0]),
            np.max(self.levels[0]),
            np.min(self.levels[1]),
            np.max(self.levels[1]),
        ]
        xx, yy, zz = np.meshgrid(self.levels[0], self.levels[1], self.levels[2])
        self.candidates = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    async def dm_sdc(self, web_client, text, channel="#asdc"):
        #              channel='DHY5REQ0H'):
        web_client.chat_postMessage(
            channel=channel,
            text=text,
            token=SDC_TOKEN,
            as_user=False,
            username="ctl",
            icon_emoji=":robot_face:",
        )

    def load_experiment_indices(self):
        # indices start at 0...
        # sqlite integer primary keys start at 1...
        df = pd.DataFrame(self.experiment_table.all())

        target_idx = self.experiment_table.count()
        experiment_idx = self.experiment_table.count(flag=False)

        return df, target_idx, experiment_idx

    def analyze_corrosion_features(self, segment=0):

        rtab = self.db.get_table("result", primary_id=False)

        for row in self.db["experiment"].all(intent="corrosion"):

            # extract features for any data that's missing
            if rtab.find_one(id=row["id"]):
                continue

            d = {"id": row["id"]}
            # V, log_I, t = load_cv(row, data_dir=self.data_dir, segment=segment)
            # cv_features, fit_data = cycvolt.analyze.model_polarization_curve(
            #     V, log_I, smooth=False, lm_method=None, shoulder_percentile=0.99
            # )

            # d.update(cv_features)
            # d['passive_region'] = d['V_tp'] - d['V_pass']

            V, I, t = load_cv(
                row, data_dir=self.data_dir, segment=segment, log=False, half=False
            )
            d["integral_current"] = np.abs(integrate.trapz(I, t))

            d["ts"] = datetime.now()
            rtab.upsert(d, ["id"])

        return

    def random_scalarization_cb(self, models, candidates, cb_beta=0.25):
        """random scalarization acquisition policy function
        depending on model likelihood, use different policy functions for different outputs
        each criterion should be framed as a minimization problem...
        """

        objective = np.zeros(candidates.shape[0])

        # sample one set of weights from a dirichlet distribution
        # that specifies our general preference on the objective weightings
        weights = stats.dirichlet.rvs(self.objective_alphas).squeeze()
        # weights = [0.0, 1.0]

        mask = None
        criteria = []
        for idx, model in enumerate(models):

            sign = self.sgn[idx]

            if idx == 0:
                # first model is the composition model -- target 15% Ni!
                criterion = composition_loss_confidence_bound(
                    model, candidates, target=0.15, sign=sign, cb_beta=cb_beta
                )

            elif model.likelihood.name in ("Gaussian", "Beta"):
                criterion = confidence_bound(
                    model, candidates, sign=sign, cb_beta=cb_beta
                )
            elif model.likelihood.name == "Bernoulli":
                criterion = classification_criterion(model, candidates)
                y_loc, _ = model.predict_y(candidates)
                mask = (y_loc > 0.5).squeeze()

            criteria.append(criterion.squeeze())

        objective = np.zeros_like(criteria[0])
        for weight, criterion in zip(weights, criteria):
            if mask is not None:
                criterion[~mask] = np.inf
            drange = np.ptp(criterion[np.isfinite(criterion)])
            criterion = (criterion - criterion.min()) / drange
            objective += weight * criterion

        return objective

    def gp_acquisition(self, resolution=100, t=0):

        df = characterization.load_characterization_results(self.db_file)

        Ni_fraction = df["NiCl2"] / (df["NiCl2"] + df["ZnCl2"])
        metal_fraction = (df["NiCl2"] + df["ZnCl2"]) / (
            df["NiCl2"] + df["ZnCl2"] + df["KCl"]
        )

        X = np.vstack((Ni_fraction, metal_fraction, df["potential"])).T
        Ni = df["Ni_ratio"].values[:, None]
        Ni_variance = df["Ni_variance"].values[:, None]
        pr = df["polarization_resistance"].values[:, None]

        # reset tf graph -- long-running program!
        gpflow.reset_default_graph_and_session()

        # set up models
        dx = 0.25 * np.ptp(self.candidates)
        models = [
            emulation.model_quality(X, Ni, dx=dx, likelihood="beta", optimize=True),
            emulation.model_property(X, Ni_variance, dx=dx, optimize=True),
            emulation.model_property(X, pr, dx=dx, optimize=True),
        ]

        # evaluate the acquisition function on a grid
        # acq = criterion.evaluate(candidates)
        acq = self.random_scalarization_cb(models, self.candidates)

        # remove previously measured candidates
        mindist = spatial.distance.cdist(X, self.candidates).min(axis=0)
        acq[mindist < 1e-5] = np.inf

        query_idx = np.argmin(acq)
        guess = self.candidates[query_idx]

        query = pd.Series(
            {"Ni_fraction": guess[0], "metal_fraction": guess[1], "potential": guess[2]}
        )
        print(query)
        return query

    @command
    async def go(self, args: str, msgdata: Dict, web_client: Any):
        """keep track of target positions and experiment list

        target and experiment indices start at 0
        sqlite integer primary keys start at 1...
        """

        previous_op = self.db["experiment"].find_one(id=self.db["experiment"].count())

        print(previous_op)

        if len(self.experiments) > 0:
            instructions = self.experiments.pop(0)
            intent = instructions[0].get("intent")
            fit_gp = False
            if intent == "deposition":
                # correctly handle "double-tap" protocol
                # assume a deposition op following another deposition is a repeat
                if previous_op is None:
                    action = Action.QUERY
                elif (self.repeat_depositions == True) and (
                    previous_op["intent"] == "deposition"
                ):
                    action = Action.REPEAT
                else:
                    action = Action.QUERY

            elif intent == "characterize":
                action = Action.PHOTONS

            elif intent == "corrosion":
                action = Action.CORRODE
        else:
            print("selecting an action")
            instructions = None
            action = select_action_single(self.db, threshold=self.coverage_threshold)
            print(action)
            # intent, fit_gp = exp_id(self.db)

        if action == Action.QUERY:
            if previous_op is not None:
                experiment_id = int(previous_op.get("experiment_id")) + 1
            else:
                experiment_id = 1
            # march through target positions sequentially
            target_idx = self.db["experiment"].count(intent="deposition")
            target = self.targets.iloc[target_idx]
            pos = {"x": target.x, "y": target.y}

        elif action == Action.REPEAT:
            experiment_id = int(previous_op.get("experiment_id"))
            # march through target positions sequentially
            target_idx = self.db["experiment"].count(intent="deposition")
            target = self.targets.iloc[target_idx]
            pos = {"x": target.x, "y": target.y}

        elif action == Action.PHOTONS:
            pos = None
            experiment_id = int(previous_op.get("experiment_id"))

        elif action == Action.CORRODE:

            if previous_op is None:
                experiment_id = 1

            elif previous_op["intent"] == "deposition":
                experiment_id = int(previous_op.get("experiment_id"))

            else:
                # if we are only doing corrosions...
                experiment_id = int(previous_op.get("experiment_id")) + 1

        # if action is Action.CORRODE, select a target without a bubble to corrode
        if action == Action.CORRODE:

            if previous_op["intent"] == "corrosion":
                count = self.db["experiment"].count(intent="corrosion")
                target = self.targets.iloc[count]

            else:

                targets = pd.DataFrame(
                    self.db["experiment"].find(experiment_id=experiment_id)
                )

                try:
                    target = targets[~(targets["has_bubble"] == True)].iloc[0]
                except KeyError:
                    target = targets.iloc[0]
            try:
                pos = {"x": target["x_combi"], "y": target["y_combi"]}
            except KeyError:
                pos = {"x": target["x"], "y": target["y"]}

        if instructions is None:

            print("get instructions")
            # get the next instruction set
            if action == Action.QUERY:
                query = self.gp_acquisition(t=experiment_id)
                instructions = deposition_instructions(
                    query, experiment_id=experiment_id
                )
            elif action == Action.REPEAT:
                instructions = json.loads(previous_op["instructions"])
                instructions = [
                    {"intent": "deposition", "experiment_id": experiment_id}
                ] + instructions
            elif action == Action.PHOTONS:
                instructions = characterize_instructions(experiment_id=experiment_id)
            elif action == Action.CORRODE:
                instructions = corrosion_instructions(experiment_id=experiment_id)

        # update the intent block to include the target position
        if pos is not None:
            instructions[0].update(pos)

        instructions[0].update({"experiment_id": experiment_id})

        print(instructions)
        # send the experiment command

        if action in {Action.QUERY, Action.REPEAT, Action.CORRODE}:
            await self.dm_sdc(
                web_client, f"<@UHT11TM6F> run_experiment {json.dumps(instructions)}"
            )
        elif action == Action.PHOTONS:
            await self.dm_sdc(
                web_client,
                f"<@UHT11TM6F> run_characterization {json.dumps(instructions)}",
            )

        return

    # @command
    # async def update(self, args: str, msgdata: Dict, web_client: Any):
    #     update_type, rest = args.split(' ', 1)
    #     print(update_type)
    #     self.update_event.set()
    #     return

    @command
    async def dm(self, args: str, msgdata: Dict, web_client: Any):
        """ echo random string to DM channel """
        dm_channel = "DHY5REQ0H"
        # dm_channel = 'DHNHM74TU'

        web_client.chat_postMessage(
            channel=dm_channel,
            text=args,
            as_user=False,
            username="ctl",
            token=CTL_TOKEN,
        )

    @command
    async def abort_running_handlers(self, args: str, msgdata: Dict, web_client: Any):
        """cancel all currently running task handlers...

        WARNING: does not do any checks on the potentiostat -- don't call this while an experiment is running...

        we could register the coroutine address when we start it up, and broadcast that so it's cancellable...?
        """

        channel = "<@UC537488J>"

        text = f"sdc: {msgdata['username']} said abort_running_handlers"
        print(text)

        # dm UC537488J (brian)
        web_client.chat_postMessage(
            channel=channel, text=text, username="ctl", token=CTL_TOKEN
        )

        current_task = asyncio.current_task()

        for task in asyncio.all_tasks():

            if task._coro == current_task._coro:
                continue

            if task._coro.__name__ == "handle":
                print(f"killing task {task._coro}")
                task.cancel()


@click.command()
@click.argument("config-file", type=click.Path())
@click.option("--verbose/--no-verbose", default=False)
def sdc_controller(config_file, verbose):

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    experiment_root, _ = os.path.split(config_file)

    # specify target file relative to config file
    target_file = config.get("target_file")
    config["target_file"] = os.path.join(experiment_root, target_file)

    data_dir = config.get("data_dir")
    if data_dir is None:
        config["data_dir"] = os.path.join(experiment_root, "data")

    figure_dir = config.get("figure_dir")
    if figure_dir is None:
        config["figure_dir"] = os.path.join(experiment_root, "figures")

    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["figure_dir"], exist_ok=True)

    if config["step_height"] is not None:
        config["step_height"] = abs(config["step_height"])

    # logfile = config.get('command_logfile', 'commands.log')
    logfile = "controller.log"
    logfile = os.path.join(config["data_dir"], logfile)

    ctl_bot = Controller(verbose=verbose, config=config, logfile=logfile)
    asyncio.run(ctl_bot.main())


if __name__ == "__main__":
    sdc_controller()
