"""Typed failures for the optional VTC ROS navigation example."""

from typing import Literal


class VtcExampleError(RuntimeError):
    """Identify which example stage failed and why."""

    def __init__(self, stage: Literal['SLAM', 'Nav2'], detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f'{stage}: {detail}')
