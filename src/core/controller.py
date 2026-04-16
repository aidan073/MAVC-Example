import numpy as np
from numpy.typing import NDArray


class Controller:
    def compute_ik(self, target_pose: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """
        Map a 6-DOF task-space target to joint positions.

        Args:
            target_pose (NDArray): [x, y, z, r, p, y] with translation in meters and
            Euler angles in radians.

        Returns:
            Joint positions in radians for each DOF.
        """
        return None
