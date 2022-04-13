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

import gpflow
from gpflowopt import acquisition
from scipy import stats
from scipy import spatial
from datetime import datetime

sys.path.append("../scirc")
sys.path.append(".")
import scirc

from asdc import slack
from asdc import analyze
from asdc import emulation
from asdc import visualization

BOT_TOKEN = open("slack_bot_token.txt", "r").read().strip()
SDC_TOKEN = open("slacktoken.txt", "r").read().strip()


def load_experiment_files(csv_files, dir="."):
    dir, _ = os.path.split(dir)
    experiments = pd.concat(
        (
            pd.read_csv(os.path.join(dir, csv_file), index_col=0)
            for csv_file in csv_files
        ),
        ignore_index=True,
    )
    return experiments


def load_experiment_json(experiment_files, dir="."):
    """ an experiment file contains a json list of experiment definitions """
    dir, _ = os.path.split(dir)

    experiments = None
    for experiment_file in experiment_files:
        with open(os.path.join(dir, experiment_file), "r") as f:
            if experiments is None:
                experiments = json.load(f)
            else:
                experiments.append(json.load(f))

    return experiments


class Controller(scirc.SlackClient):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()

    def __init__(self, config=None, verbose=False, logfile=None, token=BOT_TOKEN):
        super().__init__(verbose=verbose, logfile=logfile, token=token)
        self.command.update(super().command)
        self.msg_id = 0
        self.update_event = asyncio.Event(loop=self.loop)

        self.confirm = config.get("confirm", True)
        self.notify = config.get("notify_slack", True)
        self.data_dir = config.get("data_dir", os.getcwd())
        self.figure_dir = config.get("figure_dir", os.getcwd())

        self.db_file = os.path.join(self.data_dir, config.get("db_file", "test.db"))
        self.db = dataset.connect(f"sqlite:///{self.db_file}")
        self.experiment_table = self.db["experiment"]

        self.targets = pd.read_csv(config["target_file"], index_col=0)
        self.experiments = load_experiment_json(
            config["experiment_file"], dir=self.data_dir
        )

        # gpflowopt minimizes objectives...
        # UCB switches to maximizing objectives...
        # swap signs for things we want to minimize (everything but V_tp)
        self.objectives = ("I_p", "slope", "V_oc", "V_tp")
        self.objective_alphas = [3, 3, 2, 1]
        self.sgn = np.array([-1, -1, -1, 1])

    async def post(self, msg, ws, channel):
        # TODO: move this to the base Client class...
        response = {
            "id": self.msg_id,
            "type": "message",
            "channel": channel,
            "text": msg,
        }
        self.msg_id += 1
        await ws.send_str(json.dumps(response))

    async def dm_sdc(self, text, channel="DHY5REQ0H"):
        response = await self.slack_api_call(
            "chat.postMessage",
            data={
                "channel": channel,
                "text": text,
                "as_user": False,
                "username": "ctl",
            },
            token=SDC_TOKEN,
        )

    def load_experiment_indices(self):
        # indices start at 0...
        # sqlite integer primary keys start at 1...
        df = pd.DataFrame(self.experiment_table.all())

        target_idx = self.experiment_table.count()
        experiment_idx = self.experiment_table.count(flag=False)

        return df, target_idx, experiment_idx

    def analyze_corrosion_features(self, segment=3):

        rtab = self.db.get_table("result", primary_id=False)

        for row in self.db["experiment"].all():

            # extract features for any data that's missing
            if rtab.find_one(id=row["id"]):
                continue

            d = {"id": row["id"]}
            echem_data = pd.read_csv(
                os.path.join(self.data_dir, row["datafile"]), index_col=0
            )
            autorange = (
                echem_data["current_range"][echem_data["segment"] == segment]
                .unique()
                .size
                > 1
            )
            data = analyze.split_data(
                echem_data.to_dict(orient="list"), segment=segment
            )
            cv_features = analyze.extract_cv_features(
                data, shoulder_percentile=0.999, autorange=autorange
            )

            d.update(cv_features)
            d["ts"] = datetime.now()

            rtab.upsert(d, ["id"])

        return

    def random_scalarization_cb(self, model_wrapper, candidates, cb_beta):
        """ random scalarization upper confidence bound acquisition policy function """

        objective = np.zeros(candidates.shape[0])

        # sample one set of weights from a dirichlet distribution
        # that specifies our general preference on the objective weightings
        weights = stats.dirichlet.rvs(self.objective_alphas).squeeze()

        if self.notify:
            slack.post_message(f"sampled objective fn weights: {weights}")

        for model, weight in zip(model_wrapper.models, weights):
            mean, var = model.predict_y(candidates)
            ucb = mean + cb_beta * np.sqrt(var)
            objective += weight * ucb.squeeze()

        return objective

    def gp_acquisition(self):

        if self.notify:
            slack.post_message(f"analyzing CV features...")

        # make sure all experiments are postprocessed and have values in the results table
        self.analyze_corrosion_features()

        # load positions, compositions, and measured values from db
        df = pd.DataFrame(self.db["experiment"].all())
        r = pd.DataFrame(self.db["result"].all())

        X = df.loc[:, ("x_combi", "y_combi")].values
        Y = r.loc[:, self.objectives].values
        candidates = self.targets.values

        # set confidence bound beta
        t = X.shape[0]
        cb_beta = 0.125 * np.log(2 * t + 1)

        # set up scaling for X (if X is wafer coords...)
        # scale targets to range (-0.5, 0.5)
        scale_factor = 2 * self.targets.max().values
        X = X / scale_factor
        candidates = candidates / scale_factor

        # remove previously measured candidates
        mindist = spatial.distance.cdist(X, candidates).min(axis=0)
        candidates = candidates[mindist > 1e-3]

        # reset tf graph -- long-running program!
        gpflow.reset_default_graph_and_session()

        if self.notify:
            slack.post_message(f"fitting GP models")

        # set up models
        # don't drop the last input dimension with wafer position inputs...
        models = [
            emulation.model_ternary(
                X,
                (self.sgn * Y)[:, idx][:, None],
                reset_tf_graph=False,
                drop_last=False,
            )
            for idx in range(Y.shape[1])
        ]

        # set up multiobjective acquisition...
        # use this as a convenient model wrapper for now...
        criterion = acquisition.HVProbabilityOfImprovement(models)

        # rescale model outputs to balance objectives...
        for model in criterion.models:
            model.normalize_output = True
        criterion.root._needs_setup = True
        criterion.optimize_restarts = 1

        # fit the surrogate models
        # gpflowopt objective will optimize the full model list...
        criterion._optimize_models()

        if self.notify:
            slack.post_message(f"evaluating acquisition function")

        # evaluate the acquisition function on a grid
        # acq = criterion.evaluate(candidates)
        acq = self.random_scalarization_cb(criterion, candidates, cb_beta)

        # plot the acquisition function...
        figpath = os.path.join(self.figure_dir, f"acquisition_plot_{t}.png")
        visualization.scatter_wafer(
            candidates * scale_factor, acq, label="acquisition", figpath=figpath
        )
        if self.notify:
            slack.post_image(figpath, title=f"acquisition at t={t}")

        query_idx = np.argmax(acq)
        guess = candidates[query_idx]

        target = guess * scale_factor
        target = pd.Series({"x": target[0], "y": target[1]})

        return target

    @command
    async def go(self, ws, msgdata, args):
        """keep track of target positions and experiment list

        target and experiment indices start at 0
        sqlite integer primary keys start at 1...
        """

        # need to be more subtle here: filter experiment conditions on 'ok' or 'flag'
        # but also: filter everything on wafer_id, and maybe session_id?
        # also: how to allow cancelling tasks and adding combi spots to a queue to redo?

        target = self.gp_acquisition()

        # target_idx = self.db['experiment'].count()
        # target = self.targets.iloc[target_idx]
        print(target)

        experiment_idx = self.db["experiment"].count(flag=False)
        experiment = self.experiments[experiment_idx]
        print(experiment)

        # send the move command -- message @sdc
        self.update_event.clear()
        args = {"x": target.x, "y": target.y}
        await self.dm_sdc(f"<@UHT11TM6F> move {json.dumps(args)}")

        # wait for the ok
        # @sdc will message us with @ctl update position ...
        await self.update_event.wait()

        # the move was successful and we've had our chance to check the previous spot
        # reload the experiment in case flags have changed
        experiment_idx = self.db["experiment"].count(flag=False)
        experiment = self.experiments[experiment_idx]
        print(experiment)

        # send the experiment command
        await self.dm_sdc(f"<@UHT11TM6F> run_experiment {json.dumps(experiment)}")

        return

    @command
    async def update(self, ws, msgdata, args):
        update_type, rest = args.split(" ", 1)
        print(update_type)
        self.update_event.set()
        return

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = "DHY5REQ0H"
        # dm_channel = 'DHNHM74TU'

        response = await self.slack_api_call(
            "chat.postMessage",
            token=SDC_TOKEN,
            data={
                "channel": dm_channel,
                "text": args,
                "as_user": False,
                "username": "ctl",
            },
        )

    @command
    async def abort_running_handlers(self, ws, msgdata, args):
        """cancel all currently running task handlers...

        WARNING: does not do any checks on the potentiostat -- don't call this while an experiment is running...

        we could register the coroutine address when we start it up, and broadcast that so it's cancellable...?
        """
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

    ctl = Controller(verbose=verbose, config=config, logfile=logfile)
    ctl.run()


if __name__ == "__main__":
    sdc_controller()
