from pathlib import Path
from typing import TypedDict

from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage


class PickPlaceScenePaths(TypedDict):
    robot_prim_path: str


def _default_robot_usd_path() -> Path:
    return Path(__file__).resolve().parent / "robot" / "robot.usd"


def _spawn_dynamic_cube(
    prim_path: str,
    scale: tuple[float, float, float],
    center: tuple[float, float, float],
    mass_kg: float = 0.08,
    color: tuple[float, float, float] = (0.85, 0.2, 0.15),
) -> None:
    """Small rigid body for pick-and-place objects."""
    stage = get_current_stage()
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(mass_kg)


def create_pick_place_scene(
    *,
    robot_usd_path: str | Path | None = None,
    robot_prim_path: str = "/World/RobotArm",
) -> PickPlaceScenePaths:
    """
    Load the arm and populate table, cubes, and bins.

    Args:
        robot_usd_path (str | Path): Filesystem path to the robot USD. Defaults to ``scene/robot/robot.usd`` next to this file.
        robot_prim_path str): Stage path where the robot reference is instanced.
    """
    usd = Path(robot_usd_path) if robot_usd_path is not None else _default_robot_usd_path()
    if not usd.is_file():
        raise FileNotFoundError(f"Robot USD not found: {usd}")

    add_reference_to_stage(str(usd.resolve()), robot_prim_path)

    # TODO: Work surface
    table_z = 0.42
    table_half_t = 0.02

    # Pickable cubes on the table.
    cube_half = 0.025
    surface_z = table_z + table_half_t + cube_half
    for i, dx in enumerate((-0.08, 0.0, 0.08)):
        _spawn_dynamic_cube(
            f"/World/PickPlace/Cube_{i}",
            scale=(cube_half * 2.0,) * 3,
            center=(0.58 + dx, 0.12, surface_z),
            color=(0.9, 0.15, 0.12),
        )

    return PickPlaceScenePaths(robot_prim_path=robot_prim_path)
