"""Run the common ROS probe inside the engine's native Humble environment."""

from pathlib import Path
import sys

from usim.bridges.smoke import probe
from usim_gazebo.runner import load_configuration


if __name__ == '__main__':
    configuration, _, _ = load_configuration(Path(sys.argv[1]))
    probe(configuration, Path(sys.argv[2]))
