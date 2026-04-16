from dataclasses import dataclass

from typing import Tuple


@dataclass(frozen=True)
class RobotConfig:
    """
    Each field name is a joint name; each value is (stiffness, damping).
    """

    joint1: Tuple[float, float] = (100, 10)
    joint2: Tuple[float, float] = (100, 10)
    joint3: Tuple[float, float] = (100, 10)
    joint4: Tuple[float, float] = (100, 10)
    joint5: Tuple[float, float] = (100, 10)
    joint6: Tuple[float, float] = (100, 10)
