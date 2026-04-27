"""Differential IK controller for a 6-DOF arm in Isaac Sim.

Each call to :py:meth:`DifferentialIKController.step_towards` consumes the
articulation's runtime Jacobian (read from PhysX) and produces one joint
position-target update via a damped pseudoinverse:

    delta_q = J^T (J J^T + lambda^2 I)^-1 * pose_error
    q_target = q_current + step_size * delta_q

Convergence requires the caller to step physics between calls -- see
:py:meth:`core.robot.Robot.move_to`, which wraps the loop.

The controller assumes the EE pose target is expressed in the **same frame**
as the world Jacobian's link transforms (i.e. world frame). When the robot's
root prim is parented at the world origin with identity rotation -- the case
in this scene -- world frame == robot root frame, so callers can pass poses
straight from :py:func:`utils.transforms.convert_wrist_pose`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Tuple

from isaacsim.core.prims import Articulation


def _matrix_to_quat_wxyz(rot: NDArray[np.floating]) -> NDArray[np.float64]:
    """Rotation matrix -> scalar-first quaternion (Shepperd's method)."""
    r = np.asarray(rot, dtype=np.float64)
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_conj(q_wxyz: NDArray[np.floating]) -> NDArray[np.float64]:
    w, x, y, z = q_wxyz
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_mul(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.float64]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_to_axis_angle(q_wxyz: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert ``(w, x, y, z)`` quaternion to ``axis * angle`` 3-vector (shortest path)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, float(q[0]))
    axis = q[1:] / sin_half
    return axis * angle


def pose_to_pos_quat(
    target_pose: NDArray[np.floating],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accept a ``(4, 4)`` homogeneous transform or a length-7 ``[x, y, z, qw, qx, qy, qz]``."""
    arr = np.asarray(target_pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr[:3, 3].copy(), _matrix_to_quat_wxyz(arr[:3, :3])
    if arr.size == 7:
        flat = arr.reshape(7)
        pos = flat[:3].copy()
        quat = flat[3:].copy()
        return pos, quat / np.linalg.norm(quat)
    raise ValueError(
        "target_pose must be a (4, 4) homogeneous transform or length-7 [x, y, z, qw, qx, qy, qz]"
    )


class DifferentialIKController:
    """Single-step Jacobian-pseudoinverse IK against an Isaac Sim Articulation.

    Args:
        articulation: Initialized Articulation view (must have a valid physics
            handle, i.e. created after ``world.reset()``).
        ee_body_name: Name of the body whose pose tracks the target (e.g.
            ``"link6"`` for the Lite6 wrist).
        ik_joint_names: Names of the articulation DOFs the controller is
            allowed to move (e.g. ``("joint1", ..., "joint6")``). All other
            DOFs (gripper fingers, etc.) are left alone.
        step_size: Scalar in (0, 1] applied to the joint delta. Smaller values
            track more conservatively; 1.0 takes the full pseudoinverse step.
        pos_tol: Position error norm at or below which the EE is considered
            converged (meters).
        rot_tol: Rotation error norm (axis-angle, radians) at or below which
            the EE is considered converged.
        damping: Tikhonov damping for the pseudoinverse (rad/s). Stabilizes
            near singularities; typical values are 1e-3 to 1e-1.
        clamp_to_limits: If True, joint position targets are clipped to the
            articulation's reported DOF limits before being applied.
    """

    def __init__(
        self,
        articulation: Articulation,
        ee_body_name: str,
        ik_joint_names: Sequence[str],
        step_size: float = 0.5,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
        damping: float = 1e-2,
        clamp_to_limits: bool = True,
    ) -> None:
        self._articulation = articulation
        self._ee_body_name = ee_body_name
        self._ik_joint_names = tuple(ik_joint_names)
        self._step_size = float(step_size)
        self._pos_tol = float(pos_tol)
        self._rot_tol = float(rot_tol)
        self._damping = float(damping)
        self._clamp_to_limits = bool(clamp_to_limits)

        dof_names = list(articulation.dof_names)
        body_names = list(articulation.body_names)
        missing_joints = [n for n in self._ik_joint_names if n not in dof_names]
        if missing_joints:
            raise ValueError(
                f"IK joints {missing_joints} not in articulation DOFs {dof_names}"
            )
        if ee_body_name not in body_names:
            raise ValueError(
                f"EE body '{ee_body_name}' not in articulation bodies {body_names}"
            )

        self._ik_joint_indices: NDArray[np.int32] = np.array(
            [dof_names.index(n) for n in self._ik_joint_names], dtype=np.int32
        )
        self._ee_body_index: int = body_names.index(ee_body_name)

        # PhysX excludes the fixed base from the Jacobian's link dimension; the
        # body index of the EE must be shifted accordingly. Floating-base
        # Jacobians additionally have 6 root-velocity columns up front.
        # ``Articulation.get_jacobian_shape`` returns ``(N_links, 6, num_dof)``
        # for fixed base or ``(N_links, 6, num_dof + 6)`` for floating base.
        jac_shape = tuple(int(x) for x in np.asarray(articulation.get_jacobian_shape()).reshape(-1))
        jac_link_count = jac_shape[0]
        self._is_fixed_base = jac_link_count == articulation.num_bodies - 1
        self._ee_jacobian_index: int = (
            self._ee_body_index - 1 if self._is_fixed_base else self._ee_body_index
        )

        if self._clamp_to_limits:
            limits = np.asarray(articulation.get_dof_limits()).reshape(-1, 2)
            self._lower_limits: NDArray[np.float64] = limits[self._ik_joint_indices, 0].astype(np.float64)
            self._upper_limits: NDArray[np.float64] = limits[self._ik_joint_indices, 1].astype(np.float64)
        else:
            self._lower_limits = np.full(len(self._ik_joint_names), -np.inf, dtype=np.float64)
            self._upper_limits = np.full(len(self._ik_joint_names), np.inf, dtype=np.float64)

    @property
    def ik_joint_names(self) -> Tuple[str, ...]:
        return self._ik_joint_names

    @property
    def ik_joint_indices(self) -> NDArray[np.int32]:
        return self._ik_joint_indices

    @property
    def ee_body_index(self) -> int:
        return self._ee_body_index

    def get_ee_pose(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return current EE ``(position, quaternion_wxyz)`` in world frame."""
        # ``get_link_transforms`` returns ``(count, max_links, 7)``. We
        # access via ``_physics_view`` because the public ``Articulation`` API
        # only exposes root poses, not per-link poses.
        link_transforms = np.asarray(
            self._articulation._physics_view.get_link_transforms()
        ).reshape(-1, self._articulation.num_bodies, 7)
        ee_row = link_transforms[0, self._ee_body_index]
        pos = ee_row[0:3].astype(np.float64).copy()
        qx, qy, qz, qw = ee_row[3:7].astype(np.float64)
        quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
        quat_wxyz /= np.linalg.norm(quat_wxyz)
        return pos, quat_wxyz

    def compute_pose_error(
        self,
        target_pose: NDArray[np.floating],
    ) -> Tuple[NDArray[np.float64], float, float]:
        """Return ``(pose_error_6, pos_err_norm, rot_err_norm)``.

        ``pose_error_6`` is ``[dx, dy, dz, rx, ry, rz]`` where the rotation
        component is the axis-angle of ``q_target * q_current^-1`` (world
        frame). This is the convention expected by the world-frame Jacobian.
        """
        target_pos, target_quat = pose_to_pos_quat(target_pose)
        ee_pos, ee_quat = self.get_ee_pose()
        pos_err = target_pos - ee_pos
        rot_err = _quat_to_axis_angle(_quat_mul(target_quat, _quat_conj(ee_quat)))
        pos_err_norm = float(np.linalg.norm(pos_err))
        rot_err_norm = float(np.linalg.norm(rot_err))
        pose_err = np.concatenate([pos_err, rot_err])
        return pose_err, pos_err_norm, rot_err_norm

    def is_converged(self, pos_err_norm: float, rot_err_norm: float) -> bool:
        return pos_err_norm <= self._pos_tol and rot_err_norm <= self._rot_tol

    def step_towards(self, target_pose: NDArray[np.floating]) -> bool:
        """Apply one differential-IK update toward ``target_pose``.

        Args:
            target_pose: ``(4, 4)`` homogeneous transform of the EE target,
                or length-7 ``[x, y, z, qw, qx, qy, qz]``. Expressed in world
                frame (== robot root frame when robot is at world origin).

        Returns:
            ``True`` if the EE is already within ``pos_tol`` and ``rot_tol``
            (no command was sent); otherwise ``False``.
        """
        pose_err, pos_err_norm, rot_err_norm = self.compute_pose_error(target_pose)
        if self.is_converged(pos_err_norm, rot_err_norm):
            return True

        raw_jac = np.asarray(self._articulation.get_jacobians())
        if raw_jac.ndim == 4:
            jac_full = raw_jac[0, self._ee_jacobian_index]
        elif raw_jac.ndim == 3:
            jac_full = raw_jac[self._ee_jacobian_index]
        else:
            raise RuntimeError(
                f"Unexpected Jacobian shape from articulation: {raw_jac.shape}"
            )
        col_offset = 0 if self._is_fixed_base else 6
        jac = jac_full[:, col_offset + self._ik_joint_indices].astype(np.float64)

        lam2 = self._damping ** 2
        jjt = jac @ jac.T
        delta_q = jac.T @ np.linalg.solve(jjt + lam2 * np.eye(6), pose_err)

        current_q = (
            np.asarray(self._articulation.get_joint_positions())
            .reshape(-1)[self._ik_joint_indices]
            .astype(np.float64)
        )
        target_q = current_q + self._step_size * delta_q
        np.clip(target_q, self._lower_limits, self._upper_limits, out=target_q)

        self._articulation.set_joint_position_targets(
            positions=target_q.reshape(1, -1),
            joint_indices=self._ik_joint_indices,
        )
        return False
