"""Per-scene worker for Scene-Physics' loss-map pipeline.

The pipeline is ``scene_physics.simulation.loss_map.gen_map`` — a brute-force
sweep of the physics likelihood over the hidden object's (x, y) prior box,
returning the likelihood surface as a 2-D grid. This module assembles exactly
the arguments that function consumes for a single ``data/<exp>/<scene>``::

    gen_map(scene_usd, prior_json, truth_json, intrinsics, scene_makeup, save_dir)

i.e. the scene's ``*_physics.usdc`` / ``*_priors.json`` / ``*_truth.json``, the
experiment's camera from ``config/<exp>.json`` (the same view the dataset was
generated and occlusion-gated against), the ``*_makeup.json`` parsed into a
``Scene_Makeup``, and the scene's ``results/`` dir (where ``gen_map`` also
drops the target plot and ``point_cloud.ply``).

Meant to be launched as its own process, one per scene::

    python -m data_pipeline.loss_map --exp data-set --scene scene001

The CLI saves the surface to ``results/loss_map.npy``. Unlike inference there
is no ``--iterations``: the sweep's extent is the prior box and its resolution
is ``loss_map.FIDELITY``. ``NUM_WORLDS`` (the GPU batch size, read by
scene_physics at import) stays env-only. ``scene_physics`` is imported lazily,
in the worker only, so the launcher process never holds a CUDA context.
"""

from __future__ import annotations

import argparse
import json

from data_pipeline.utils.config import load_experiment_config
from data_pipeline.utils.paths import data_dir

__all__ = ["scene_args", "run_loss_map"]


def scene_args(exp: str, scene: str) -> tuple:
    """Positional args for ``gen_map`` for one ``data/<exp>/<scene>``.

    Returns ``(scene_usd, prior_json, truth_json, intrinsics, scene_makeup,
    save_dir)`` — splat straight into ``gen_map(*scene_args(...))``.

    Raises ``FileNotFoundError`` naming any of the four required inputs (physics
    USD / priors / truth / makeup) the scene is missing.
    """
    from scene_physics.properties.structs import Scene_Makeup

    data = data_dir(exp) / scene / "data"
    physics = data / f"{scene}_physics.usdc"
    priors = data / f"{scene}_priors.json"
    truth = data / f"{scene}_truth.json"
    makeup = data / f"{scene}_makeup.json"

    missing = [p.name for p in (physics, priors, truth, makeup) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{exp}/{scene}: missing {', '.join(missing)} in {data} "
            f"— build the scene first (gen_scene)."
        )

    results = data_dir(exp) / scene / "results"
    results.mkdir(parents=True, exist_ok=True)

    camera = load_experiment_config(exp).camera
    mk = json.loads(makeup.read_text())
    scene_makeup = Scene_Makeup(
        static=mk["static"], observed=mk["observed"], hidden=mk["hidden"]
    )

    return (
        str(physics),
        str(priors),
        str(truth),
        camera,
        scene_makeup,
        str(results),
    )


def run_loss_map(exp: str, scene: str):
    """Generate the loss map for one ``data/<exp>/<scene>``; return the 2-D
    likelihood surface over the hidden object's (x, y) prior box.

    Thin bridge: builds the call via :func:`scene_args` and splats it into
    ``gen_map``. ``scene_physics`` is imported here, lazily, so importing this
    module costs no CUDA context until the work runs.
    """
    from scene_physics.simulation.loss_map import gen_map

    return gen_map(*scene_args(exp, scene))


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate a Scene-Physics loss map for one data/<exp>/<scene>.",
    )
    ap.add_argument(
        "--exp", help="experiment id, e.g. data-set (config/<exp>.json must exist)",
        required=True
        )

    ap.add_argument(
        "--scene", help="scene name, e.g. scene001",
        required=True
        )
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    result = run_loss_map(args.exp, args.scene)

    import numpy as np

    out = data_dir(args.exp) / args.scene / "results" / "loss_map.npy"
    np.save(out, result)
    print(f"Saved {result.shape} loss map to {out}")
