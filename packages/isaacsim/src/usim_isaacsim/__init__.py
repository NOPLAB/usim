"""Isaac engine execution; native modules load only when a simulation starts."""

from usim.simulation import SimulationConfig, wheel_velocities as wheel_velocities
from usim_isaacsim.sim import IsaacSimulator


def simulate(configuration: SimulationConfig, *, stop=None) -> None:
    """Run a configured robot scene with optional ROS I/O."""
    IsaacSimulator().run(configuration, stop=stop)
