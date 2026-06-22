"""Per-scene worker for Scene-Physics' inference (importance-sampling) pipeline.

The pipeline is ``scene_physics.simulation.sim_sampling.run_importance_sampling``
— physics-based importance sampling that infers the hidden object's placement.
This module assembles exactly the arguments that function consumes for a single
``data/<exp>/<scene>``::

    run_importance_sampling(scene_usd, prior_json, truth_json,
                            intrinsics, scene_makeup, save_dir, iterations)

i.e. the scene's ``*_physics.usdc`` / ``*_priors.json`` / ``*_truth.json``, the
experiment's camera from ``config/<exp>.json`` (the same view the dataset was
generated and occlusion-gated against), the ``*_makeup.json`` parsed into a
``Scene_Makeup``, and the scene's ``results/`` dir.

Meant to be launched as its own process, one per scene::

    python -m data_pipeline.inference exp01 scene001
    python -m data_pipeline.inference exp01 scene001 --iterations 100

``--iterations`` (Gibbs iterations) defaults to the ``NUM_EPOCHS`` env var
(scene_physics default 50); ``NUM_WORLDS`` (the GPU batch size, read by
scene_physics at import) stays env-only. ``scene_physics`` is imported lazily, in
the worker only, so the launcher process never holds a CUDA context.
"""

from __future__ import annotations

import argparse
import json
import os

from data_pipeline.utils.config import load_experiment_config
from data_pipeline.utils.paths import data_dir

__all__ = ["scene_args", "run_inference"]


def scene_args(exp: str, scene: str, *, iterations: int | None = None) -> tuple:
    """Positional args for ``run_importance_sampling`` for one ``data/<exp>/<scene>``.

    Returns ``(scene_usd, prior_json, truth_json, intrinsics, scene_makeup,
    save_dir, iterations)`` — splat straight into
    ``run_importance_sampling(*scene_args(...))``. ``iterations`` defaults to the
    ``NUM_EPOCHS`` env var (scene_physics default 50).

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

    if iterations is None:
        iterations = int(os.environ.get("NUM_EPOCHS", 50))

    return (
        str(physics),
        str(priors),
        str(truth),
        camera,
        scene_makeup,
        str(results),
        iterations,
    )


def run_inference(exp: str, scene: str, *, iterations: int | None = None):
    """Run importance sampling for one ``data/<exp>/<scene>``; return its
    ``(scene, model, likelihood)``.

    Thin bridge: builds the call via :func:`scene_args` and splats it into
    ``run_importance_sampling``. ``scene_physics`` is imported here, lazily, so
    importing this module costs no CUDA context until the work runs.
    """
    from scene_physics.simulation.sim_sampling import run_importance_sampling

    return run_importance_sampling(*scene_args(exp, scene, iterations=iterations))


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Scene-Physics importance sampling on one data/<exp>/<scene>.",
    )
    ap.add_argument("--exp", help="experiment id, e.g. exp01 (config/<exp>.json must exist)")
    ap.add_argument("--scene", help="scene name, e.g. scene001")
    ap.add_argument(
        "--iterations", "-n", type=int, default=None,
        help="Gibbs iterations (default: $NUM_EPOCHS, else 50)",
    )
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_inference(args.exp, args.scene, iterations=args.iterations)
