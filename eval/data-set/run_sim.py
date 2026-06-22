import argparse
from pathlib import Path

from pxr import Usd, UsdPhysics

from scene_physics.simulation.simulation import run_simulation


# The physics-enabled USDs authored by to_physics.py live one per scene at
# data/data-set/<scene>/data/<scene>_physics.usdc -- not flat in data/data-set/, and not the
# render-only manual/ exports (those carry no UsdPhysics schemas, so add_usd imports
# nothing and the sim is empty). Key each scene by its folder and bind it to the
# *_physics.usdc inside, the same glob gen_truth.py and render.py use.
EXP_DIR = Path(__file__).resolve().parents[2] / "data" / "data-set"
SCENES = {
    usd.parent.parent.name: usd
    for usd in sorted(EXP_DIR.glob("scene*/data/*_physics.usdc"))
}

def has_rigid_bodies(usd_path):
    """True if the USD authors at least one dynamic body. Raw Blender exports
    mislabeled as *_physics.usdc import as plain meshes with no UsdPhysics schemas, so
    add_usd brings in geometry but nothing moves and the sim renders empty. The flat,
    name-based glob can't tell those apart from real physics USDs -- only the schemas can."""
    stage = Usd.Stage.Open(str(usd_path))
    return any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Declare scene u want rendered"
    )
    ap.add_argument("--scene", default=next(iter(SCENES), None), choices=sorted(SCENES))

    args = ap.parse_args()
    if not SCENES:
        ap.error(f"no scene*/data/*_physics.usdc under {EXP_DIR}")

    usd = SCENES[args.scene]
    if not has_rigid_bodies(usd):
        ap.error(
            f"{usd.relative_to(EXP_DIR.parents[1])} has no UsdPhysics rigid bodies "
            f"(likely a raw export mislabeled as _physics.usdc) -- re-author it with: "
            f"uv run eval/data-set/to_physics.py --scene {args.scene}"
        )

    run_simulation(str(usd), "h")
