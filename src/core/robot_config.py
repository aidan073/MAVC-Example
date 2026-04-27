from dataclasses import dataclass

from typing import Tuple


@dataclass(frozen=True)
class RobotConfig:
    """
    Each field name is a joint name; each value is (stiffness, damping).
    """

    joint1: Tuple[float, float] = (1.0e5, 5.0e3)
    joint2: Tuple[float, float] = (1.0e5, 5.0e3)
    joint3: Tuple[float, float] = (1.0e5, 5.0e3)
    joint4: Tuple[float, float] = (5.0e4, 2.5e3)
    joint5: Tuple[float, float] = (5.0e4, 2.5e3)
    joint6: Tuple[float, float] = (2.0e4, 1.0e3)
