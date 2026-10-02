"""Optional Isaac library port; importing it does not load Isaac or ROS."""

from usim.simulation import SimulationConfig, wheel_velocities as wheel_velocities
from usim.ports.isaac.sim import IsaacSimulator


def simulate(configuration: SimulationConfig, *, stop=None) -> None:
    """Run a mobile robot using default Isaac scene paths and optional ROS I/O."""
    IsaacSimulator().run(configuration, stop=stop)
