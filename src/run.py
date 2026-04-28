"""Drive a Franka Panda EE with live MAVC commands over TCP via IsaacLab Diff IK.

This script mirrors ``ref.py`` (the IsaacLab differential-IK tutorial) for
scene/controller setup, but the goal is no longer a hard-coded list of poses
that cycle every 150 ticks. Instead, a :class:`mavc_receiver.Receiver` listens
for binary ``Command`` frames; each one is converted from camera-frame
``palm_position`` + ``palm_orientation`` to a root-frame
``[x, y, z, qw, qx, qy, qz]`` IK target via
:func:`utils.transforms.camera_xyzrpy_to_root_pose7`.

Run with::

    ./isaaclab.sh -p src/run.py

Then point a MAVC-Sender at ``<host>:<port>`` (default ``0.0.0.0:9000``).
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MAVC-Example live diff-IK demo (Franka Panda).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--mavc_host", type=str, default="0.0.0.0", help="MAVC-Receiver bind host.")
parser.add_argument("--mavc_port", type=int, default=9000, help="MAVC-Receiver bind port.")
parser.add_argument(
    "--mavc_reach",
    type=float,
    default=0.7,
    help=(
        "Meters of max reach. Multiplies the normalized shoulder-frame "
        "palm_position from each Command. e.g. ``palm_position=(1, 0, 0)`` with "
        "reach=0.6 means 'wrist 0.6 m along the shoulder-frame +X axis'."
    ),
)
parser.add_argument(
    "--shoulder_xyz",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=[0.0, 0.0, 0.0],
    help=(
        "Where the operator's shoulder maps to in the robot root frame (meters). "
        "Each Command's palm_position is wrist-relative-to-shoulder; this offset "
        "is added to the axis-swapped result so a zero palm_position lands the EE "
        "target at this point."
    ),
)
parser.add_argument(
    "--receive_hz",
    type=float,
    default=2.0,
    help=(
        "Max rate (Hz) at which the MAVC receive queue is drained. The "
        "simulation loop only calls ``rx.spin_once()`` when at least "
        "``1 / receive_hz`` seconds have elapsed since the last drain. "
        "Set to <= 0 to drain on every physics tick (no throttle)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from mavc_receiver import Command, Receiver
from mavc_receiver.cfg_parser import ReceiverCfg

from utils.transforms import camera_xyzrpy_to_root_pose7


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Single-arm tabletop scene (Franka Panda)."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop with live MAVC commands."""
    robot = scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Reset the arm to its default joint state so we have a known starting pose.
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    scene.write_data_to_sim()

    # Initialize the IK command to the EE's current root-frame pose so the arm
    # holds station until the first MAVC command arrives.
    sim_dt = sim.get_physics_dt()
    scene.update(sim_dt)
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    root_pose_w = robot.data.root_pose_w
    init_pos_b, init_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:, 0:3] = init_pos_b
    ik_commands[:, 3:7] = init_quat_b
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)

    # === MAVC-Receiver wiring ===========================================
    # Every Command updates ``ik_commands`` in place; the next IK ``compute()``
    # picks up the new target. The callback runs on the main thread inside
    # ``rx.spin_once()``, so no locking is required.
    rx_cfg = ReceiverCfg(bind_host=args_cli.mavc_host, bind_port=args_cli.mavc_port)
    rx = Receiver(rx_cfg)

    shoulder_origin_in_robot = tuple(float(v) for v in args_cli.shoulder_xyz)

    def on_command(_rx: Receiver, cmd: Command) -> None:
        px, py, pz = cmd.palm_position
        roll, pitch, yaw = cmd.palm_orientation
        scale = float(args_cli.mavc_reach)
        # palm_position is shoulder-frame normalized coords (origin at the
        # shoulder, axes parallel to the camera frame). Just scale to meters.
        shoulder_xyz_m = (px * scale, py * scale, pz * scale)
        target7 = camera_xyzrpy_to_root_pose7(
            *shoulder_xyz_m,
            roll,
            pitch,
            yaw,
            shoulder_origin_in_robot=shoulder_origin_in_robot,
        )
        target_tensor = torch.tensor(target7, device=ik_commands.device, dtype=ik_commands.dtype)
        ik_commands[:, 0:7] = target_tensor
        diff_ik_controller.reset()
        diff_ik_controller.set_command(ik_commands)
        print(
            f"[MAVC-Example] seq={cmd.sequence_id} "
            f"shoulder_xyz_m=({shoulder_xyz_m[0]:+.3f},{shoulder_xyz_m[1]:+.3f},{shoulder_xyz_m[2]:+.3f}) "
            f"rpy=({roll:+.3f},{pitch:+.3f},{yaw:+.3f}) "
            f"-> root_pos=({target7[0]:+.3f},{target7[1]:+.3f},{target7[2]:+.3f}) "
            f"root_quat_wxyz=({target7[3]:+.3f},{target7[4]:+.3f},{target7[5]:+.3f},{target7[6]:+.3f}) "
            f"grip={cmd.grip_amount:.2f}"
        )

    rx.register_callback(on_command, execute_on_spin=True)
    rx.run()
    print(f"[MAVC-Example] MAVC-Receiver listening on {rx_cfg.bind_host}:{rx_cfg.bind_port} (plain TCP)")

    receive_hz = float(args_cli.receive_hz)
    spin_period = 1.0 / receive_hz if receive_hz > 0.0 else 0.0
    last_spin_time = 0.0

    try:
        while simulation_app.is_running():
            # Drain any pending Commands (each spin_once dequeues at most one),
            # but only after ``spin_period`` seconds have elapsed since the last
            # drain. Setting --receive_hz <= 0 reverts to draining every tick.
            now = time.monotonic()
            if now - last_spin_time >= spin_period:
                while rx.spin_once():
                    pass
                last_spin_time = now

            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # apply actions
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            scene.update(sim_dt)

            # update marker positions
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
    finally:
        rx.stop()


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
