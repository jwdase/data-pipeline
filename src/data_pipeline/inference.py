"""Per-scene worker for Scene-Physics' model pipeline.

The model pipeline is ``scene_physics.simulation.sim_sampling.run_importance_sampling``
— physics-based importance sampling that infers the hidden object's placement.
This module builds the arguments it needs for a single ``data/<exp>/<scene>``
(the scene's USD / priors / truth / makeup, the experiment's camera from
``config/<exp>.json`` — the same view the dataset was generated and
occlusion-gated against — and the ``results/`` dir its outputs go to) and runs it.

It is meant to be launched as its own process, one per scene, by
``eval/<exp>/run_sim.py``::

    python -m data_pipeline.inference exp01 scene001

``NUM_EPOCHS`` (Gibbs iterations) and ``NUM_WORLDS`` (read by ``scene_physics`` at
import) come from the environment. ``scene_physics`` is imported lazily, in the
worker only, so the launcher process never holds a CUDA context.
"""

from __future__ import annotations

import json
import os

from data_pipeline.utils.config import load_experiment_config
from data_pipeline.utils.paths import data_dir

__all__ = ["scene_args"]


def scene_args(exp: str, scene: str, *, iterations: int | None = None) -> tuple:
    """Positional args for ``run_importance_sampling`` for one ``data/<exp>/<scene>``.

    Returns ``(scene_usd, priors, truth, camera, scene_makeup, results_dir,
    iterations)`` — splat straight into ``run_importance_sampling(*scene_args(...))``.
    ``iterations`` defaults to the ``NUM_EPOCHS`` env var (scene_physics default 50).
    """
    from scene_physics.properties.structs import Scene_Makeup

    data = data_dir(exp) / scene / "data"
    results = data_dir(exp) / scene / "results"
    results.mkdir(parents=True, exist_ok=True)

    camera = load_experiment_config(exp).camera
    mk = json.loads((data / f"{scene}_makeup.json").read_text())
    scene_makeup = Scene_Makeup(
        static=mk["static"], observed=mk["observed"], hidden=mk["hidden"]
    )

    if iterations is None:
        iterations = int(os.environ.get("NUM_EPOCHS", 50))

    return (
        str(data / f"{scene}_physics.usdc"),
        str(data / f"{scene}_priors.json"),
        str(data / f"{scene}_truth.json"),
        camera,
        scene_makeup,
        str(results),
        iterations,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m data_pipeline.inference <exp> <scene>")
        raise SystemExit(1)

    from scene_physics.simulation.sim_sampling import run_importance_sampling

    run_importance_sampling(*scene_args(sys.argv[1], sys.argv[2]))
