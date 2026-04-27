import numpy as np

# Frame mapping matrix for camera frame to robot frame (X->X, Y->-Z, Z->-Y)
M = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=float)

# Frame mapping matrix for EE frame to robot frame (X->Z, Y->-X, Z->-Y)
M2 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # <-- was missing parentheses


def rpy_to_matrix_zyx(roll, pitch, yaw):
    # TODO: MAVC-Sender should just send the rotation matrix so that this isn't needed
    """Reconstruct rotation matrix from ZYX Euler angles (same convention as deconstruction in MAVC-Sender)."""
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


def convert_wrist_pose(position_camera, roll_cam, pitch_cam, yaw_cam):
    """
    Convert wrist position and orientation from camera frame to robot frame,
    accounting for the difference between the observed wrist frame and the
    robot's expected end effector frame.

    Args:
        position_camera:         (x, y, z) position of wrist in camera frame
        roll_cam, pitch_cam, yaw_cam: ZYX Euler angles in camera frame (radians)

    Returns:
        T: 4x4 homogeneous transformation matrix in robot root frame,
           with the orientation expressed in the EE frame convention
    """
    # --- Position ---
    position_robot = M @ np.array(position_camera)

    # --- Orientation ---
    R_camera = rpy_to_matrix_zyx(roll_cam, pitch_cam, yaw_cam)
    R_robot = M @ R_camera @ M  # camera -> robot root frame (M == M.T)
    R_ee = R_robot @ M2  # robot root frame -> EE frame convention

    # --- Assemble 4x4 homogeneous transform ---
    T = np.eye(4)
    T[:3, :3] = R_ee
    T[:3, 3] = position_robot

    return T
