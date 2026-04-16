from core.controller import Controller
from core.robot_config import RobotConfig

import numpy as np
from dataclasses import fields
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.controllers.articulation_controller import ArticulationController

from numpy.typing import NDArray
from typing import Iterator, Tuple


class Robot:
    """Composes `RobotConfig` and `Controller`; owns the Articulation instance used to drive the Robot prim."""

    def __init__(
        self,
        config: RobotConfig,
        articulation: Articulation,
        controller: Controller | None = None,
    ) -> None:
        self._config = config
        self._articulation = articulation
        self._controller = controller if controller is not None else Controller()

        self._articulation_controller = ArticulationController()
        self._articulation_controller.initialize(articulation)

    @property
    def config(self) -> RobotConfig:
        return self._config

    @property
    def controller(self) -> Controller:
        """IK controller."""
        return self._controller

    @property
    def articulation(self) -> Articulation:
        """Isaac Sim articulation view for this robot prim."""
        return self._articulation

    @property
    def articulation_controller(self) -> ArticulationController:
        """Low-level joint command interface used after physics is running."""
        return self._articulation_controller

    def move_to(
        self,
        target_pose: NDArray[np.floating],
    ) -> None:
        """
        Move joints to necessary positions based on a target End-Effector pose.

        Args:
            target_pose (NDArray): [x, y, z, r, p, y] with translation in meters and
            Euler angles in radians.
        """
        target_joint_positions = self._compute_joint_position_targets(target_pose=target_pose)
        self._apply_joint_positions(joint_positions=target_joint_positions)

    def iter_joint_gains(self) -> Iterator[Tuple[str, float, float]]:
        """Yield ``(joint_name, stiffness, damping)`` from the composed config."""
        for f in fields(self._config):
            stiffness, damping = getattr(self._config, f.name)
            yield f.name, stiffness, damping

    def _apply_pd_gains(self) -> None:
        """Set implicit PD Kp/Kd on this articulation for joints named in the config."""
        table = {name: (kp, kd) for name, kp, kd in self.iter_joint_gains()}
        names: list[str] = []
        kps: list[float] = []
        kds: list[float] = []
        for dof in self._articulation.dof_names:
            if dof in table:
                kp, kd = table[dof]
                names.append(dof)
                kps.append(kp)
                kds.append(kd)
        if not names:
            return
        kps_arr = np.array([kps], dtype=np.float32)
        kds_arr = np.array([kds], dtype=np.float32)
        self._articulation.set_gains(kps=kps_arr, kds=kds_arr, joint_names=names)

    def _compute_joint_position_targets(
        self,
        target_pose: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Map a 6-DOF task-space target to joint position targets.

        Args:
            target_pose (NDArray): [x, y, z, r, p, y] with translation in meters and
            Euler angles in radians.

        Returns:
            Joint positions in radians for each DOF.
        """

        joint_targets = self._controller.compute_ik(target_pose)
        if joint_targets is None:
            return np.array(self._articulation.get_joint_positions(), copy=True)
        joint_targets = np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        n_dof = int(self._articulation.num_dof)
        if joint_targets.size != n_dof:
            raise ValueError(f"IK returned {joint_targets.size} values but articulation has {n_dof} DOFs")
        return joint_targets

    def _apply_joint_positions(self, joint_positions: NDArray[np.floating]) -> None:
        """Send a position command for all DOFs on the next physics step."""
        action = ArticulationAction(joint_positions=np.asarray(joint_positions, dtype=np.float64))
        self._articulation_controller.apply_action(action)
