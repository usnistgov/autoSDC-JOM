# SDC Client

## Overview

`python asdc/client.py ${EXPERIMENT_DIR}/config.yaml` - start the sdc client.

The client program will read the [configuration](#configuration) file and connect to the `#asdc` channel
using slack's [real-time messaging API](https://api.slack.com/rtm).

`asdc` uses a (mostly) json api to mediate control between computer programs and humans.
These commands are primarily intended to be used programmatically, but we've
found them to be pretty useful for free-form debugging as well.

The SDC client will parse all messages in the `#asdc` channel for commands,
which are begin by @mentioning the SDC bot account
and consist of a command name and a json payload of arguments.
The SDC client will also respond to direct messages (the @mention is still required).

## SDC json commands

### move

slack command for horizontal translation of the SDC sample stage ([implementation][asdc.client.SDC.move]). Direct z-stage access is not allowed.

This slack command wraps [move_stage][asdc.client.SDC.move_stage], and will lift up the cell by the [configured step height](#stage-configuration) for translating the stage horizontally and returning to the initial z-stage setting.

**Example:** align wafer coordinate `(0,0)` with the camera

`#!json @sdc move {"x": 0.0, "y": 0.0, "reference_frame": "camera"}`

**Parameters**

| Name              | Type                              | Description                                                   | Default  |
|-------------------|-----------------------------------|---------------------------------------------------------------|----------|
| `x`               | `float`                           | wafer x coordinate (mm)                                       |          |
| `y`               | `float`                           | wafer y coordinate (mm)                                       |          |
| `reference_frame` | `Enum['cell', 'camera', 'laser]]` | reference frame to align with the specified wafer coordinates | `'cell'` |


!!! warning
    The `move` command does not currently perform input validation.
    `move` does not currently know about any physical obstructions within the throw of the stages,
    and will happily ram into things if directed to do so...


### stop_pumps

::: asdc.localclient.SDC.stop_pumps

**Example:**

`#!json @sdc stop_pumps`


### run_experiment

Execute some SDC experiment protocol.


### droplet

debugging command for droplet formation subroutine.

If `fill_time` or `shrink_time` are not specified, `droplet` will request input
on the terminal for each of these phases (i.e. press `<ENTER>` when ready),
and helpfully report timings for each step in a slack message.

**Example:** ask the `droplet` routine to time things for us

`#!json @sdc droplet {"shrink_rate": 1.3, "fill_rate": 0.7, "cleanup": 15}`

**Example:** test a fully-specified droplet formation routine

`#!json @sdc droplet {"shrink_rate": 1.3, "fill_rate": 0.7, "fill_time": 19, "shrink_time": 2, "cleanup": 15}`

**Parameters**

| Name             | Type  | Description                                         | Default |
|------------------|-------|-----------------------------------------------------|---------|
| `prep_height`    | float | z setting to grow the droplet (meters)              | 0.004 (4mm) |
| `wetting_height` | float | z setting to wet the droplet to the surface         | 0.0011  (1.1mm) |
| `fill_rate`      | float | counterpumping ratio during droplet growth          |    0.75 |
| `fill_time`      | float | droplet growth duration (s)                         |    None |
| `shrink_rate`    | float | counterpumping ratio during droplet wetting phase   |     1.1 |
| `shrink_time`    | float | droplet wetting duration (s)                        |    None |
| `flow_rate`      | float | total flow rate during droplet formation (mL/min)   |     0.5 |
| `cleanup`        | float | duration of pre-droplet-formation cleanup siphoning |       0 |
| `stage_speed`    | float | stage velocity during droplet formation op          |   0.001 |



### reflectance
slack command to collect a reflectance linescan ([implementation][asdc.client.SDC.reflectance]).

!!! warning
    The `reflectance` command translates the sample stage
    (see [reflectance_linescan][asdc.client.SDC.reflectance_linescan]).
    Ensure that the z-stage is such that the cell is not in contact
    with the sample to avoid dragging, which could potentially damage
    the sample or the cell.


### comment

slack command to provide unstructured feedback on an experiment and store it directly in a string field in the database.

Since this command is not intended for programmatic use, it uses a (brittle) positional string format of the form
`#!bash comment ${primary_key} ${comment_text}`. Invoking the `comment` command multiple times for a given experiment
will append subsequent comments to the comment field in the database. (refer to the [docs][asdc.client.SDC.comment] for more precision)

**Example:** record feedback on experiment 1

`#!json @sdc comment 1 we observed something strange with this deposition`



### checkpoint


### flag


### coverage


## SDC Client implementation

::: asdc.localclient.SDC


## configuration

#### input/output files

| Name              | Type      | Description                                                              | Default        |
|-------------------|-----------|--------------------------------------------------------------------------|----------------|
| `target_file`     | `os.path` | csv file containing wafer coordinates, relative to `${EXPERIMENT_DIR}/`. | `map.csv`      |
| `data_dir`        | `os.path` | data directory, relative to `${EXPERIMENT_DIR}`.                         | `data/`        |
| `figure_dir`      | `os.path` | figure directory, relative to `${EXPERIMENT_DIR}`.                       | `figures/`     |
| `db_file`         | `os.path` | sqlite database file, relative to `data_dir`.                            | `sdc.db`       |
| `command_logfile` | `os.path` | log file for slack command history, relative to `data_dir`               | `commands.log` |


#### output configuration

| Name           | Type   | Description                  | Default |
|----------------|--------|------------------------------|---------|
| `notify_slack` | `bool` | post status updates to slack | `False` |
| `plot_cv`      | `bool` | post CV plot to slack        | `False` |
| `plot_current` | `bool` | post current plot to slack   | `False` |


#### debugging options
Here is a collection of debug flags that toggles debug functionality

| Name                 | Type                     | Description                                              | Default    |
|----------------------|--------------------------|----------------------------------------------------------|------------|
| `test`               | `bool`                   |                                                          | `False`    |
| `test_cell`          | `bool`                   | if set, replace potentiostat actions with a no-op.       | `False`    |
| `cell`               | `INTERNAL` or `EXTERNAL` | which cell to use                                        | `INTERNAL` |
| `confirm`            | `bool`                   | ask for permission before performing potentiostat action | `True`     |
| `confirm_experiment` | `bool`                   |                                                          | `False`    |

#### quality checks
| Name                 | Type    | Description                                                                   | Default |
|----------------------|---------|-------------------------------------------------------------------------------|---------|
| `coverage_threshold` | `float` | minimum coverage for a "good" deposit. (default                               | `0.9`)  |
| `current_threshold`  | `float` | average current heuristic for detecting "failed" electrodepositions. (default | `1e-5`) |

#### adafruit configuration
| Name            | Type  | Description                         | Default |
|-----------------|-------|-------------------------------------|---------|
| `adafruit_port` | `str` | serial port for the microcontroller | `COM9`  |


#### solutions handling
| Name              | Type                          | Description                                                  | Default      |
|-------------------|-------------------------------|--------------------------------------------------------------|--------------|
| `solutions`       | `Dict[str, Dict[str, float]]` | dictionary mapping syringe pumps to solution compositions.   |              |
| `pump_array_port` | `str`                         | serial port for the pump array                               | `COM10`      |
| `cleanup_pause`   | `float`                       | time delay for solution cleanup after an e-chem op. (default | `0` seconds) |

#### stage configuration
| Name                    | Type              | Description                                                                          | Default                 |
|-------------------------|-------------------|--------------------------------------------------------------------------------------|-------------------------|
| `initial_combi_position | Dict[str, float]` | manual specification of wafer coordinates for the initial reference frame alignment. | `{"x":  0.0, "y": 0.0}` |
| `frame_orientation`     | (default          | wafer orientation relative to the lab frame.                                         | `-y`                    |
| `stage_ip               | str`              | IP address of stage controller                                                       | `192.168.10.11`         |
| `speed                  | float`            | stage speed in m/s.                                                                  | `1e-3`, i.e. `1 mm/s`)  |
| `step_height            | float`            | step height in meters (default                                                       | `0.0011`                |

#### controller-specific options
| Name             | Type | Description                                      | Default             |
|------------------|------|--------------------------------------------------|---------------------|
| `experiment_file | str` | json file for predefined experimental protocol   | `instructions.json` |
| `domain_file     | str` | json file defining active learning design space. | `domain.json`       |
