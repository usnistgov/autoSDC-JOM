import json
import inspect
import logging
from typing import Optional
from dataclasses import dataclass
from collections.abc import Iterable

logger = logging.getLogger(__name__)


def standard_hydrogen_ref(potential):
    """convert standard hydrogen potentials to Ag/AgCl reference"""
    return potential + 0.197


class SDCArgs:
    """base class for default experiment arguments

    relies on a bit of introspection magic to generate param strings for the Ametek backend
    the backend wants a comma-delimited argument string

    subclass SDCargs with a dataclass to specify argument defaults (and ordering)

    use a thin subclass of these dataclasses to make the interface nicer
    e.g. the interface subclass can take `versus` as a single argument
    and pass that value to `versus_initial` and `versus_final`

    """

    @property
    def name(self):
        return type(self).__name__

    @classmethod
    def from_dict(cls, args):
        """ override default dataclass arguments, but skip any keys that aren't attributes of the dataclass """
        values = {
            k: v for k, v in args.items() if k in inspect.signature(cls).parameters
        }

        voltage_overrides = [k for k, v in values.items() if v == "VS SHE"]
        for k in voltage_overrides:
            values[k] = "VS REF"
            _, voltage_key = k.split("_")
            vkey = f"{voltage_key}_potential"
            values[vkey] = standard_hydrogen_ref(values[vkey])

        return cls(**values)

    def format(self):
        """ format a comma-delimited argument string in the order expected by Ametek backend """
        paramstring = ",".join(
            [str(arg).upper() for name, arg in self.__dict__.items()]
        )
        logger.debug(f"running {self.name}: {paramstring}")
        return paramstring

    def as_dict(self):
        return self.__dict__

    def setup(self, pstat_experiments):
        """ a bit hacky -- needs a reference to the .NET library for potentiostat setup funcs """
        args = self.getargs()
        setup_func = getattr(pstat_experiments, self.setup_func)
        status = setup_func(args)
        return args

    def update_relative_scan(self, open_circuit_potential):
        """Mutating update -- transform potentials 'VS HOLD' to 'VS REF'.

        potentials currently have matching keys like
        {key}_potential = value
        versus_{key} = reference_type_string

        this is a bit hacky. cleaner would be to have types for potentials:

        class Potential(pydantic.BaseModel):
            value: float = 0.0
            reference: pydantic.Literal["VS REF", "VS OC", "VS PREV", "VS HOLD"]

        then it should be safer to make `str(potential) -> "0.0,VS OC"`
        """
        if open_circuit_potential is None:
            return

        # update any potential that is relative to "VS HOLD"
        to_update = [key for key, value in self.as_dict().items() if value == "VS HOLD"]

        for versus_key in to_update:

            # key matching (brittle / depends on field names in experiment arguments)
            key = versus_key.replace("versus_", "")
            potential_key = f"{key}_potential"

            # transform potential specified relative to measured OCP
            # to the reference electrode frame
            relative_potential = getattr(self, potential_key)
            absolute_potential = open_circuit_potential + relative_potential

            # write back data to both potential and reference fields
            setattr(self, potential_key, absolute_potential)
            setattr(self, versus_key, "VS REF")

        return


@dataclass
class MeasureOCP(SDCArgs):
    """ overload setup method -- this experiment takes no arguments """

    setup_func: str = "AddMeasureOpenCircuit"

    def setup(self, pstat_experiments):
        setup_func = getattr(pstat_experiments, self.setup_func)
        status = setup_func()
        return None


class SDCChain:
    """ wrapper class to set up sequenced experiments """

    def __init__(self, *experiments, remeasure_ocp=False):

        self.remeasure_ocp = remeasure_ocp

        if len(experiments) == 1 and isinstance(experiments[0], Iterable):
            self.experiments = experiments[0]
        else:
            self.experiments = experiments

    def setup(self, pstat_experiments):
        args = []
        for e in self.experiments:
            _args = e.setup(pstat_experiments)
            args.append(_args)
            if self.remeasure_ocp:
                MeasureOCP().setup(pstat_experiments)


@dataclass
class LPRArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = "VS REF"
    final_potential: float = 1.0
    versus_final: str = "VS REF"
    step_height: float = 0.1
    step_time: float = 0.1
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "AUTO"
    acquisition_mode: str = "AUTO"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    leave_cell_on: str = "YES"
    cell: str = "EXTERNAL"
    enable_ir_compensation: str = "DISABLED"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"


@dataclass
class PotentiostaticArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = "VS REF"
    time_per_point: float = 0.00001
    duration: float = 10
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "AUTO"
    acquisition_mode: str = "AUTO"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    leave_cell_on: str = "YES"
    cell: str = "EXTERNAL"
    enable_ir_compensation: str = "DISABLED"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"


@dataclass
class PotentiodynamicArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = "VS REF"
    final_potential: float = 0.0
    versus_final: str = "VS REF"
    step_height: float = 0.1
    step_time: float = 1.0
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "AUTO"
    acquisition_mode: str = "AUTO"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    leave_cell_on: str = "YES"
    cell: str = "EXTERNAL"
    enable_ir_compensation: str = "DISABLED"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"


@dataclass
class LSVArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = "VS REF"
    final_potential: float = 0.65
    versus_final: str = "VS REF"
    scan_rate: float = 1.0
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "AUTO"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    leave_cell_on: str = "YES"
    cell: str = "EXTERNAL"
    enable_ir_compensation: str = "DISABLED"
    user_defined_the_amount_of_ir_comp: float = 1
    use_previously_determined_ir_comp: str = "YES"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"


@dataclass
class StaircaseLSVArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = "VS REF"
    final_potential: float = 0.65
    versus_final: str = "VS REF"
    step_height: float = 0.1
    step_time: float = 1.0
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "AUTO"
    acquisition_mode: str = "AUTO"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    leave_cell_on: str = "NO"
    cell_to_use: str = "INTERNAL"
    enable_ir_compensation: str = "DISABLED"
    user_defined_the_amount_of_ir_comp: str = 1
    use_previously_determined_ir_comp: str = "YES"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"


@dataclass
class TafelArgs(SDCArgs):
    initial_potential: float = -0.25
    versus_initial: str = "VS OC"
    final_potential: float = 0.25
    versus_final: str = "VS OC"
    step_height: float = 0.001
    step_time: float = 0.5
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "AUTO"
    acquisition_mode: str = "AUTO"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    leave_cell_on: str = "YES"
    cell: str = "EXTERNAL"
    enable_ir_compensation: str = "DISABLED"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"


@dataclass
class PotentiostaticEISArgs(SDCArgs):
    """Potentiostatic EIS

    point_spacing can be `LINEAR` or `LOGARITHMIC`
    n_points: frequency samples. in logarithmic scale, this is points per decade
    data_quality: value in (0, 10)
    measurement_delay: delay between impedance samples

            "initial_potential": 0.0,
            "versus": "OC",
            "start_frequency": 1e5,
            "end_frequency": 1e-2,
            "amplitude_potential": 0.02,
            "point_spacing": "LOGARITHMIC",
            "n_points": 5,
    """

    start_frequency: float = 1e5  # Hz
    end_frequency: float = 1e-2  # Hz
    amplitude_potential: float = 0.02  # V
    point_spacing: str = "LOGARITHMIC"
    n_points: int = 5
    data_quality: int = 1
    measurement_delay: float = 0.1
    initial_potential: float = -0.25
    versus_initial: str = "VS OC"
    current_range: str = "2MA"
    electrometer: str = "AUTO"
    leave_cell_on: str = "YES"
    cell: str = "EXTERNAL"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"


@dataclass
class OpenCircuitArgs(SDCArgs):
    time_per_point: float = 1
    duration: float = 10
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "2MA"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    cell: str = "EXTERNAL"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"
    e_resolution: str = "AUTO"


@dataclass
class CorrosionOpenCircuitArgs(SDCArgs):
    time_per_point: float = 1
    duration: float = 10
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "2MA"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    cell: str = "EXTERNAL"
    bandwidth: str = "AUTO"
    low_current_interface_bandwidth: str = "AUTO"
    e_resolution: str = "AUTO"


@dataclass
class CyclicVoltammetryArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = "VS REF"
    vertex_1_potential: float = 1.0
    versus_vertex_1: str = "VS REF"
    vertex_1_hold: float = 0
    acquire_data_during_vertex_hold_1: str = "NO"
    vertex_2_potential: float = -1.0
    versus_vertex_2: str = "VS REF"
    vertex_2_hold: float = 0
    acquire_data_during_vertex_hold_2: str = "NO"
    scan_rate: float = 0.1
    cycles: int = 3
    limit_1_type: Optional[str] = None
    limit_1_direction: str = "<"
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = "<"
    limit_2_value: float = 0
    current_range: str = "AUTO"
    electrometer: str = "AUTO"
    e_filter: str = "AUTO"
    i_filter: str = "AUTO"
    leave_cell_on: str = "YES"
    cell: str = "EXTERNAL"
    enable_ir_compensation: str = "DISABLED"
    user_defined_the_amount_of_ir_comp: float = 1
    use_previously_determined_ir_comp: str = "YES"
    bandwidth: str = "AUTO"
    final_potential: float = 0.0
    versus_final: str = "VS REF"
    low_current_interface_bandwidth: str = "AUTO"
