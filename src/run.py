from isaacsim import SimulationApp

# hide_ui: hides main Kit chrome (menus, docks, outliner, etc.). It does not strip
# viewport-local HUD, guides, or viewport extension menubars -- see demo_viewport.
simulation_app = SimulationApp({"headless": False, "hide_ui": True})

from core.robot import Robot
from core.controller import DifferentialIKController
from core.robot_config import RobotConfig
from scene.create_scene import load_pick_place_scene
from utils.transforms import convert_wrist_pose

from pathlib import Path
import numpy as np
from isaacsim.core.api import World
import omni.kit.viewport.utility as vu
from isaacsim.core.prims import Articulation

script_dir = Path(__file__).resolve().parent
SCENE_PATH = script_dir / "scene" / "assets" / "pick_place_scene.usd"
ROBOT_PRIM_PATH = "/World/lite6"
CAMERA_PRIM_PATH = "/World/Camera"

EE_BODY_NAME = "link6"
IK_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# Controller convergence settings -- tweak if poses are missed.
IK_STEP_SIZE = 0.5
IK_POS_TOL = 1e-3       # meters
IK_ROT_TOL = 1e-2       # radians (~0.57 deg)
IK_DAMPING = 1e-2
MAX_STEPS_PER_POSE = 300

CAMERA_FRAME_TEST_MESSAGES: list[tuple[float, float, float, float, float, float]] = [
    (0.30, 0.10, 0.40, 0.0, 0.0, 0.0),
    # (0.35, -0.05, 0.40, 0.1, -0.15, 0.2),
    # (0.25, 0.15, 0.40, -0.05, 0.1, -0.3),
    # (0.30, 0.10, 0.50, 0.0, 0.0, 0.0),
]

# Targets expressed directly in robot/world frame, so they are independent of
# ``convert_wrist_pose``. Use these to verify the differential-IK controller
# in isolation. Identity orientation == link6 axes aligned with world axes.
def _identity_pose(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


ROBOT_FRAME_TEST_TARGETS: list[np.ndarray] = [
    _identity_pose(0.30, 0.0, 0.30),
    _identity_pose(0.25, 0.10, 0.25),
    _identity_pose(0.20, -0.10, 0.35),
]


def move_robot_from_camera_xyzrpy(
    robot: Robot,
    world: World,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    max_steps: int = MAX_STEPS_PER_POSE,
) -> bool:
    """Pretend one MAVC message arrived: position + ZYX rpy in camera frame."""
    T = convert_wrist_pose([x, y, z], roll, pitch, yaw)
    return robot.move_to(T, world=world, max_steps=max_steps)


def run_camera_frame_message_tests(
    robot: Robot,
    world: World,
    max_steps_per_pose: int = MAX_STEPS_PER_POSE,
) -> None:
    """Apply each synthetic camera-frame message and report convergence."""
    for i, (x, y, z, roll, pitch, yaw) in enumerate(CAMERA_FRAME_TEST_MESSAGES):
        print(f"[MAVC-Example] === Camera-frame test {i} ===")
        robot.log_current_joint_positions(label=f"camera test {i} pre")
        converged = move_robot_from_camera_xyzrpy(
            robot, world, x, y, z, roll, pitch, yaw, max_steps=max_steps_per_pose
        )
        robot.log_current_joint_positions(label=f"camera test {i} post")
        status = "converged" if converged else "did NOT converge"
        print(f"[MAVC-Example] camera test {i}: {status}")


def run_robot_frame_target_tests(
    robot: Robot,
    world: World,
    max_steps_per_pose: int = MAX_STEPS_PER_POSE,
) -> None:
    """Drive the EE to each ``ROBOT_FRAME_TEST_TARGETS`` pose (no frame conversion)."""
    for i, T in enumerate(ROBOT_FRAME_TEST_TARGETS):
        print(f"[MAVC-Example] === Robot-frame test {i} ===")
        robot.log_current_joint_positions(label=f"robot test {i} pre")
        converged = robot.move_to(T, world=world, max_steps=max_steps_per_pose)
        robot.log_current_joint_positions(label=f"robot test {i} post")
        status = "converged" if converged else "did NOT converge"
        print(f"[MAVC-Example] robot test {i}: {status}")


def main() -> None:
    load_pick_place_scene(SCENE_PATH)
    vp_api, _vp_window = vu.get_active_viewport_and_window()
    vp_api.camera_path = CAMERA_PRIM_PATH

    world = World()
    arm = Articulation(prim_paths_expr=ROBOT_PRIM_PATH, name="pick_place_arm")
    world.scene.add(arm)
    world.reset()

    controller = DifferentialIKController(
        articulation=arm,
        ee_body_name=EE_BODY_NAME,
        ik_joint_names=IK_JOINT_NAMES,
        step_size=IK_STEP_SIZE,
        pos_tol=IK_POS_TOL,
        rot_tol=IK_ROT_TOL,
        damping=IK_DAMPING,
    )
    robot = Robot(config=RobotConfig(), articulation=arm, controller=controller)

    # Let the arm settle into its rest pose before issuing IK commands.
    for _ in range(60):
        world.step(render=True)

    # Run the robot-frame tests first to validate the differential-IK controller
    # in isolation, then exercise the camera-frame pipeline.
    run_robot_frame_target_tests(robot, world)
    run_camera_frame_message_tests(robot, world)

    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    try:
        import traceback
        import logging

        main()
    except Exception:
        traceback.print_exc()
        logging.exception("Unhandled exception in main")
        raise
    finally:
        simulation_app.close()
