# Protocol

`asdc` uses a json protocol to define experiments and actions, to support logging, experiment composition, and communication between multiple processes/machines.

## electrochemistry

### potentiostatic

Apply a constant potential for a fixed interval.

**Parameters**

| Name            | Type    | Description                | Default |
|-----------------|---------|----------------------------|---------|
| `potential`     | `float` | applied potential (V)      | None    |
| `duration`      | `float` | operation duration (s)     | 10      |
| `current_range` | `str`   | current range to use       | 'AUTO'  |

Example:

```json
{
    "op": "potentiostatic",
    "potential": 0.5,
    "duration": 30
}
```

### LSV

Linear scan voltammetry -- sweep the potential while measuring current.

**Parameters**

| Name                | Type                    | Description                                      | Default |
|---------------------+-------------------------+--------------------------------------------------+---------|
| `initial_potential` | `float`                 | starting potential (V)                           | None    |
| `final_potential`   | `float`                 | ending potential (V)                             | None    |
| `scan_rate`         | `float`                 | scan rate (V/s)                                  | None    |
| `current_range`     | `str`                   | current range to use                             | 'AUTO'  |

Example:

```json
{
    "op": "lsv",
    "initial_potential": 0.0,
    "final_potential": 1.0,
    "scan_rate": 0.075
}
```

### LPR
(Linear) Polarization Resistance -- sweep the potential with a step curve, while measuring current.

**Parameters**

| Name                | Type                    | Description                                      | Default |
|---------------------+-------------------------+--------------------------------------------------+---------|
| `initial_potential` | `float`                 | starting potential (V)                           | None    |
| `final_potential`   | `float`                 | ending potential (V)                             | None    |
| `vs`                | Enum['VS REF', 'VS OC'] | potential relative to reference electrode or OCP | 'VS OC' |
| `step_size`         | `float`                 | potential step height (V)                        | None    |
| `step_time`         | `float`                 | potential step length (s)                        | None    |
| `current_range`     | `str`                   | current range to use                             | 'AUTO'  |


Example:

```json
{
    "op": "lpr",
    "initial_potential": 0.0,
    "final_potential": 1.0,
    "step_size": 0.1,
    "step_time": 0.1,
    "vs": "VS OC"
}
```



### CV

Multi-cycle cyclic voltammetry -- apply a triangular wave potential pattern and measure current.

**Parameters**

| Name                 | Type    | Description                              | Default |
|----------------------|---------|------------------------------------------|---------|
| `initial_potential`  | `float` | starting potential (V)                   | None    |
| `vertex_potential_1` | `float` | first vertex potential in the cycle (V)  | None    |
| `vertex_potential_2` | `float` | second vertex potential in the cycle (V) | None    |
| `initial_potential`  | `float` | final potential (V)                      | None    |
| `scan_rate`          | `float` | operation duration (s)                   | None    |
| `cycles`             | `int`   | operation duration (s)                   | None    |
| `current_range`      | `str`   | post current plot to slack               | 'AUTO'  |

Example:

```json
{
    "op": "cv",
    "initial_potential": 0.0,
    "vertex_potential_1": -1.0,
    "vertex_potential_2": 1.2,
    "final_potential": 0.0,
    "scan_rate": 0.075,
    "cycles": 2
}
```
