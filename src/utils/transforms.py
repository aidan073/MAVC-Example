"""Human-shoulder-frame wrist pose -> robot-root-frame EE pose for MAVC commands.

The MAVC sender transmits the human wrist's pose relative to the human
shoulder. We map this to a target pose for the robot wrist (panda_hand) in
the robot root frame -- which we treat as the robot-shoulder frame -- via
the chain:

    R_rw^rs = (M_rs_hs @ R_hw^hs @ M_rs_hs.T) @ M_rw_hw

    p_rs    = M_rs_hs @ p_hs + shoulder_origin_in_robot

The orientation step has two pieces:

    (i)  ``M_rs_hs @ R_hw^hs @ M_rs_hs.T`` is the *similarity transform*
         that re-expresses the human's wrist rotation in the robot-shoulder
         basis. If the human rotates their wrist by some angle around some
         HS axis, this gives the equivalent rotation in RS by the same
         angle around the corresponding RS axis (``M_rs_hs @ axis_hs``).
         Simple left-multiplication ``M_rs_hs @ R_hw^hs`` would be wrong --
         that's frame composition through nested frames, not a basis
         change.

    (ii) Post-multiplying by ``M_rw_hw`` applies the static offset between
         the human wrist's neutral pose and the panda_hand's neutral pose.
         At neutral input (``R_hw^hs == I``) the chain collapses to
         ``R_rw^rs == M_rw_hw``, which is exactly the value read off Isaac
         Sim's Property tab when the gripper is sitting at its visual
         home pose.

Where:
    R_hw^hs : human wrist in the human-shoulder basis, built from the
              received roll/pitch/yaw (ZYX intrinsic Euler).
    M_rs_hs : human shoulder expressed in the robot-shoulder basis -- a
              fixed axis swap that encodes "the operator stands facing the
              robot, head up".
    M_rw_hw : robot wrist expressed in the human-wrist basis -- equivalent
              to the panda_hand world orientation at neutral input. Tuned
              via XYZ-intrinsic Euler degrees below.
"""

from typing import Sequence, Tuple

import numpy as np


# Human shoulder expressed in robot-shoulder basis (= the rotation that maps
# a vector from human-shoulder coords to robot-shoulder coords). Columns are
# the human-shoulder basis vectors expressed in robot-shoulder coords:
#   HS  +X (operator's right)             -> RS (0, 1, 0) = +Y (robot's left)
#   HS  +Y (down)                         -> RS (0, 0,-1) = -Z (down)
#   HS  +Z (forward, away from operator)  -> RS (-1,0, 0) = -X (toward operator)
# Consistent with the operator standing in front of the robot, facing it.
M_rs_hs = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ],
    dtype=float,
)


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


# Robot wrist expressed in the human-wrist basis. When the sender transmits
# ``palm_orientation == (0, 0, 0)`` (i.e. R_hw^hs == I), this is the
# rotation the robot wrist ends up at relative to the human wrist -- it
# absorbs the difference between "human wrist palm-down, fingers forward"
# and "panda_hand neutral pose". Read off Isaac Sim's Property tab
# (XYZ-intrinsic degrees) and tweak here until the gripper visually faces
# the right direction at the neutral pose.
ROBOT_WRIST_WRT_HUMAN_WRIST_EULER_XYZ_DEG: Tuple[float, float, float] = (
    175.731,
    -89.031,
    -6.616,
)
M_rw_hw = _euler_xyz_intrinsic_to_matrix(ROBOT_WRIST_WRT_HUMAN_WRIST_EULER_XYZ_DEG)


def rpy_to_matrix_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Reconstruct rotation matrix from ZYX Euler angles (sender-side convention)."""
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
    position_shoulder: Sequence[float],
    roll: float,
    pitch: float,
    yaw: float,
    shoulder_origin_in_robot: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Human-shoulder-frame wrist pose -> 4x4 transform in robot root frame.

    Implements the chain documented at the top of this module:

        R_rw^rs = (M_rs_hs @ R_hw^hs @ M_rs_hs.T) @ M_rw_hw
        p_rs    = M_rs_hs @ p_hs + shoulder_origin_in_robot

    Args:
        position_shoulder: Wrist ``(x, y, z)`` in the human-shoulder frame
            (meters).
        roll, pitch, yaw: Wrist orientation as ZYX intrinsic Euler angles
            in the human-shoulder frame (radians).
        shoulder_origin_in_robot: Where the human shoulder lives in the
            robot root frame ``(x, y, z)`` (meters). Added after the
            ``M_rs_hs`` axis swap so the operator can be placed anywhere
            in the robot's workspace without re-tuning ``M_rs_hs``.

    Returns:
        ``(4, 4)`` homogeneous transform whose rotation block is the target
        EE orientation in the robot root frame and whose translation block
        is the target EE position in the robot root frame -- exactly what
        an IsaacLab :class:`DifferentialIKController` consumes when its
        end-effector body is ``panda_hand``.
    """
    p_hs = np.asarray(position_shoulder, dtype=float)
    p_rs = M_rs_hs @ p_hs + np.asarray(shoulder_origin_in_robot, dtype=float)

    R_hw_hs = rpy_to_matrix_zyx(roll, pitch, yaw)
    # Similarity transform: re-express the human wrist rotation in the
    # robot-shoulder basis (same angle, axis transformed by M_rs_hs).
    R_motion_rs = M_rs_hs @ R_hw_hs @ M_rs_hs.T
    # Then apply the static neutral-pose offset.
    R_rw_rs = R_motion_rs @ M_rw_hw

    T = np.eye(4)
    T[:3, :3] = R_rw_rs
    T[:3, 3] = p_rs
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
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    shoulder_origin_in_robot: Sequence[float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float, float, float, float, float]:
    """Human-shoulder ``(x, y, z, r, p, y)`` -> root-frame ``(x, y, z, qw, qx, qy, qz)``.

    Uses :func:`convert_wrist_pose` for the SE(3) mapping, then converts the
    rotation block to a wxyz quaternion. Output layout matches what an
    IsaacLab :class:`DifferentialIKController` consumes when
    ``command_type='pose'``.
    """
    T = convert_wrist_pose([x, y, z], roll, pitch, yaw, shoulder_origin_in_robot)
    pos = T[:3, 3]
    quat_wxyz = matrix_to_quat_wxyz(T[:3, :3])
    out = np.concatenate([pos, quat_wxyz]).astype(np.float64)
    return tuple(float(v) for v in out)  # type: ignore[return-value]
