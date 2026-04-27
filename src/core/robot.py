from core.controller import DifferentialIKController
from core.robot_config import RobotConfig

import numpy as np
from dataclasses import fields
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation

from numpy.typing import NDArray
from typing import Iterator, Tuple


class Robot:
    """Composes ``RobotConfig`` and a differential-IK ``Controller`` for the Lite6.

    The articulation is driven via implicit PD targets sourced from the IK
    controller. ``move_to`` runs a closed-loop convergence: each iteration
    asks the controller for one Jacobian-pseudoinverse step, then advances
    the world one physics tick so the joints can chase the new target.
    """

    def __init__(
        self,
        config: RobotConfig,
        articulation: Articulation,
        controller: DifferentialIKController,
    ) -> None:
        self._config = config
        self._articulation = articulation
        self._controller = controller
        self._apply_pd_gains()

    @property
    def config(self) -> RobotConfig:
        return self._config

    @property
    def controller(self) -> DifferentialIKController:
        return self._controller

    @property
    def articulation(self) -> Articulation:
        return self._articulation

    def move_to(
        self,
        target_pose: NDArray[np.floating],
        world: World,
        max_steps: int = 300,
        render: bool = True,
        verbose: bool = True,
    ) -> bool:
        """Drive the EE toward ``target_pose`` via differential IK.

        Args:
            target_pose: ``(4, 4)`` homogeneous transform of the EE target in
                world frame (== robot root frame when the robot is at the
                world origin), or length-7 ``[x, y, z, qw, qx, qy, qz]``.
            world: Isaac Sim ``World`` instance used to advance physics
                between IK iterations.
            max_steps: Maximum number of (IK update + physics step) iterations
                to run before giving up. The controller's ``pos_tol`` and
                ``rot_tol`` decide convergence inside this budget.
            render: Forwarded to ``world.step``; set ``False`` for headless
                runs that don't need a render per tick.
            verbose: When True, log start/end EE pose and final error.

        Returns:
            ``True`` if the controller reported convergence within
            ``max_steps``, else ``False``.
        """
        if verbose:
            self._log_target_vs_current("start", target_pose)
        for _ in range(max_steps):
            converged = self._controller.step_towards(target_pose)
            if converged:
                if verbose:
                    self._log_target_vs_current("converged", target_pose)
                return True
            world.step(render=render)
        _, pos_err, rot_err = self._controller.compute_pose_error(target_pose)
        if verbose:
            self._log_target_vs_current("max_steps", target_pose)
        return self._controller.is_converged(pos_err, rot_err)

    def _log_target_vs_current(self, tag: str, target_pose: NDArray[np.floating]) -> None:
        from core.controller import pose_to_pos_quat

        ee_pos, ee_quat = self._controller.get_ee_pose()
        target_pos, target_quat = pose_to_pos_quat(target_pose)
        _, pos_err, rot_err = self._controller.compute_pose_error(target_pose)
        print(
            f"[MAVC-Example] move_to[{tag}]: "
            f"ee_pos={np.array2string(ee_pos, precision=4)}, "
            f"target_pos={np.array2string(target_pos, precision=4)}, "
            f"pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad"
        )

    def iter_joint_gains(self) -> Iterator[Tuple[str, float, float]]:
        for f in fields(self._config):
            stiffness, damping = getattr(self._config, f.name)
            yield f.name, stiffness, damping

    def _apply_pd_gains(self) -> None:
        """Set implicit PD Kp/Kd on this articulation for joints named in the config.

        Writes the runtime drive gains via ``set_gains``; the Isaac Sim
        Property tab reads USD-authored values, so it will keep showing the
        original numbers even though these are what PhysX actually uses.
        """
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

    def log_current_joint_positions(self, label: str = "") -> None:
        """Print live joint positions for all DOFs. Call after stepping physics."""
        try:
            current = np.asarray(self._articulation.get_joint_positions()).reshape(-1)
        except Exception as e:
            print(f"[MAVC-Example] could not read current joint positions: {e}")
            return
        readback = ", ".join(
            f"{dof}={float(current[i]):.6f}"
            for i, dof in enumerate(self._articulation.dof_names)
            if i < current.size
        )
        prefix = "[MAVC-Example] current joint positions"
        if label:
            prefix += f" ({label})"
        print(f"{prefix} (rad): {readback}")
