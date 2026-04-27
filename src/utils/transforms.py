"""Camera-frame -> robot-root-frame pose conversion for MAVC commands."""

from typing import Sequence, Tuple

import numpy as np

# Camera frame -> robot root frame axis swap (X->X, Y->-Z, Z->-Y).
M = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=float)


def _euler_xyz_intrinsic_to_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    """Build a rotation matrix from XYZ-intrinsic Euler angles in degrees.

    Matches Isaac Sim's Property-tab convention (``xformOp:rotateXYZ``):
    apply rotation about the body X axis, then the body Y axis, then the
    body Z axis. Result is ``R = Rx(a) @ Ry(b) @ Rz(c)``.
    """
    a, b, c = np.deg2rad(np.asarray(angles_deg, dtype=float))
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


# Visual orientation of ``panda_hand`` (in Isaac Sim's Property tab, XYZ-intrinsic
# degrees) that matches the wrist's neutral / identity pose as expressed in the
# camera frame. When the sender transmits ``palm_orientation = (0, 0, 0)``, the
# resulting EE target rotation in robot-root frame is exactly this orientation.
# Tweak in degrees here to retune the wrist convention without touching code below.
EE_NEUTRAL_EULER_XYZ_DEG: Tuple[float, float, float] = (-3.921, -79.335, 178.048)

# Robot root frame -> EE-frame convention. M2 is post-multiplied onto the
# camera-rotated frame inside :func:`convert_wrist_pose`, so it equals R_ee
# when ``palm_orientation == (0, 0, 0)``.
M2 = _euler_xyz_intrinsic_to_matrix(EE_NEUTRAL_EULER_XYZ_DEG)


def rpy_to_matrix_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Reconstruct rotation matrix from ZYX Euler angles (camera-side convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def convert_wrist_pose(
    position_camera: Sequence[float],
    roll_cam: float,
    pitch_cam: float,
    yaw_cam: float,
) -> np.ndarray:
    """Camera-frame wrist pose -> 4x4 homogeneous transform in robot root frame.

    The output rotation is expressed in the MAVC EE-frame convention (i.e. with
    ``M2`` post-applied), matching what an IsaacLab IK controller expects when
    its target is the gripper body (``panda_hand`` for Franka).
    """
    position_robot = M @ np.array(position_camera, dtype=float)

    R_camera = rpy_to_matrix_zyx(roll_cam, pitch_cam, yaw_cam)
    R_robot = M @ R_camera @ M
    R_ee = R_robot @ M2

    T = np.eye(4)
    T[:3, :3] = R_ee
    T[:3, 3] = position_robot
    return T


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> scalar-first quaternion ``(w, x, y, z)`` (Shepperd's method)."""
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


def camera_xyzrpy_to_root_pose7(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> Tuple[float, float, float, float, float, float, float]:
    """Camera-frame ``(x, y, z, roll, pitch, yaw)`` -> root-frame ``(x, y, z, qw, qx, qy, qz)``.

    Uses :func:`convert_wrist_pose` for the SE(3) mapping, then converts the
    rotation block to a wxyz quaternion. Matches the input layout IsaacLab's
    :class:`DifferentialIKController` expects when ``command_type='pose'``.
    """
    T = convert_wrist_pose([x, y, z], roll, pitch, yaw)
    pos = T[:3, 3]
    quat_wxyz = matrix_to_quat_wxyz(T[:3, :3])
    out = np.concatenate([pos, quat_wxyz]).astype(np.float64)
    return tuple(float(v) for v in out)  # type: ignore[return-value]
