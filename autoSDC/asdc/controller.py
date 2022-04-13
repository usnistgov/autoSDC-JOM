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

sys.path.append("../scirc")
sys.path.append(".")
import scirc

from asdc import slack

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

        self.targets = pd.read_csv(
            config["target_file"], index_col=0, dtype={"x": np.float, "y": np.float}
        )
        self.experiments = load_experiment_json(
            config["experiment_file"], dir=self.data_dir
        )

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

    def get_next_experiment(self, experiment_idx):

        if len(self.experiments) == 1:
            experiment = self.experiments[0]
        else:
            experiment = self.experiments[experiment_idx]

        return experiment

    @command
    async def go(self, ws, msgdata, args):
        """keep track of target positions and experiment list

        target and experiment indices start at 0
        sqlite integer primary keys start at 1...
        """

        # need to be more subtle here: filter experiment conditions on 'ok' or 'flag'
        # but also: filter everything on wafer_id, and maybe session_id?
        # also: how to allow cancelling tasks and adding combi spots to a queue to redo?

        target_idx = self.db["experiment"].count()
        target = self.targets.iloc[target_idx]
        print(target)

        experiment_idx = self.db["experiment"].count(flag=False)
        experiment = self.get_next_experiment(experiment_idx)
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
        experiment = self.get_next_experiment(experiment_idx)
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
