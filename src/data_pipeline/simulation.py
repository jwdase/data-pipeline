"""Re-simulate one scene's physics USD with Scene-Physics' XPBD solver.

Two Scene-Physics entry points, both loading the physics USD authored at
``data/<exp>/<scene>/data/<scene>_physics.usdc`` and re-settling it with
``scene_gen``'s exact solver setup (gravity / up-axis / substeps / dt). They
differ only in output, so this module dispatches between them by whether a
recording is requested::

    run_simulation(scene_usd, _)         # interactive GL viewer only (2nd arg unused)
    run_simulation_save(scene_usd, out)  # + writes a USD recording of the sim

A raw Blender export mislabeled as ``*_physics.usdc`` carries no UsdPhysics
schemas, so ``add_usd`` imports geometry but no bodies and the sim is empty —
guarded here; re-author such a scene with ``to_physics`` first.

    python -m data_pipeline.simulation data-set scene050
    python -m data_pipeline.simulation data-set scene050 --output sim.usdc

``scene_physics`` is imported lazily, so importing this module holds no GPU
context until the sim actually runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdPhysics

from data_pipeline.utils.paths import data_dir

__all__ = ["scene_usd_path", "has_rigid_bodies", "run_sim"]


def scene_usd_path(exp: str, scene: str) -> Path:
    """Path to the physics USD both simulators load:
    ``data/<exp>/<scene>/data/<scene>_physics.usdc``."""
    return data_dir(exp) / scene / "data" / f"{scene}_physics.usdc"


def has_rigid_bodies(usd_path) -> bool:
    """True if the USD authors at least one dynamic body. A raw Blender export
    mislabeled as ``*_physics.usdc`` imports as plain meshes with no UsdPhysics
    schemas, so ``add_usd`` brings in geometry but nothing moves and the sim
    renders empty — only the schemas tell the two apart."""
    stage = Usd.Stage.Open(str(usd_path))
    return any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())


def run_sim(exp: str, scene: str, *, output: str | Path | None = None) -> None:
    """Re-simulate ``data/<exp>/<scene>``'s physics USD.

    ``output is None`` → interactive GL viewer only (``run_simulation``).
    ``output`` a path → also write a USD recording there (``run_simulation_save``).

    Resolves the scene's ``*_physics.usdc`` and checks it carries rigid bodies
    (else the sim would be silently empty) before handing the path — the only
    input these Scene-Physics functions need — to the right one.
    """
    usd = scene_usd_path(exp, scene)
    if not usd.exists():
        avail = sorted(
            p.parent.parent.name
            for p in data_dir(exp).glob("scene*/data/*_physics.usdc")
        )
        raise FileNotFoundError(
            f"{exp}/{scene}: no physics USD at {usd}.\n"
            f"available in {exp}: {', '.join(avail) or '(none)'}"
        )
    if not has_rigid_bodies(usd):
        raise ValueError(
            f"{usd} has no UsdPhysics rigid bodies (raw export mislabeled as "
            f"_physics.usdc) — the sim would be empty; re-author it with to_physics."
        )

    if output is None:
        from scene_physics.simulation.simulation import run_simulation

        run_simulation(str(usd), None)  # 2nd positional arg is unused by Scene-Physics
    else:
        from scene_physics.simulation.simulation import run_simulation_save

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_simulation_save(str(usd), str(out))


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Re-simulate one scene's physics USD with Scene-Physics' XPBD solver.",
    )
    ap.add_argument("--exp", help="experiment id, e.g. data-set")
    ap.add_argument("--scene", help="scene name, e.g. scene050")
    ap.add_argument(
        "--output", "-o", metavar="PATH", default=None,
        help="write a USD recording of the sim here (run_simulation_save); "
             "omit for an interactive viewer only (run_simulation).",
    )
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_sim(args.exp, args.scene, output=args.output)
