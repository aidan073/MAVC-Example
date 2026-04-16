from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "anti_aliasing": 0})

from core.robot import Robot
from core.robot_config import RobotConfig
from scene.create_scene import create_pick_place_scene

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation


def main() -> None:
    world = World.instance()
    world.scene.add_default_ground_plane()

    scene_paths = create_pick_place_scene()
    robot_prim_path = scene_paths["robot_prim_path"]

    world.reset()

    arm = Articulation(prim_paths_expr=robot_prim_path, name="pick_place_arm")
    robot = Robot(RobotConfig(), arm)

    while simulation_app.is_running():
        world.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
