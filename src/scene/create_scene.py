from pathlib import Path

from isaacsim.core.utils.stage import open_stage


def load_pick_place_scene(scene_path: Path) -> None:
    """Open the pick/place scene USD on the active stage."""
    scene_path = Path(scene_path).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Scene USD not found at provided path: {scene_path}")
    # USD / Kit bindings expect str paths, not pathlib.Path (Boost.Python rejects PosixPath).
    open_stage(str(scene_path))
