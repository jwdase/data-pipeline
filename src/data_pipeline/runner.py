"""Datagen runner — the thin bridge from an experiment to Scene-Physics.

A ``data_gen/<exp>/`` script only has to describe *what* to generate (a
``SceneSpec``) and *how much* (scene count); this module resolves the
experiment's camera from ``config/<exp>.json``, points the output at
``data/<exp>/``, and calls ``scene_physics.data_gen.generate_dataset_parallel``
(single-GPU batched oversample-and-filter). The heavy lifting — sampling,
settling, occlusion gating, artifact writing — all lives in Scene-Physics.
"""

from __future__ import annotations

from scene_physics.data_gen import SceneSpec, generate_dataset_parallel

from data_pipeline.utils.config import load_experiment_config
from data_pipeline.utils.paths import data_dir

__all__ = ["run_datagen"]


def run_datagen(
    exp: str,
    spec: SceneSpec,
    n_scenes: int,
    *,
    num_worlds: int | None = None,
    seed: int = 0,
    start_index: int = 1,
    max_candidates: int | None = None,
):
    """Generate ``n_scenes`` scenes for experiment ``exp`` into ``data/<exp>/``.

    The experiment's camera view comes from ``config/<exp>.json``; everything
    else is passed straight through to ``generate_dataset_parallel``. ``seed``
    makes a run reproducible — vary it (with the same or a fresh ``out_root``)
    to draw independent additional scenes.

    ``num_worlds`` is the GPU batch size. Leave it ``None`` to defer to
    scene_physics' ``DEFAULT_BATCH_WORLDS`` — i.e. the ``NUM_WORLDS`` env var
    (default 5) — so ``NUM_WORLDS=64 uv run ...`` is honored; pass an int to
    override the env explicitly.
    """
    config = load_experiment_config(exp)
    out_root = data_dir(exp, create=True)

    if num_worlds is None:
        # Defer to scene_physics' env-driven default instead of hardcoding a
        # number, so the NUM_WORLDS env var actually takes effect.
        from scene_physics.data_gen.scene_gen import DEFAULT_BATCH_WORLDS
        num_worlds = DEFAULT_BATCH_WORLDS

    print(
        f"[run_datagen] exp={exp} target={spec.target} n_scenes={n_scenes} "
        f"num_worlds={num_worlds} seed={seed}\n"
        f"             camera fov={config.camera.fov_degree}deg "
        f"{config.camera.width}x{config.camera.height} eye={config.camera.eye.tolist()}\n"
        f"             out_root={out_root}"
    )

    return generate_dataset_parallel(
        spec,
        n_scenes=n_scenes,
        out_root=out_root,
        num_worlds=num_worlds,
        seed=seed,
        intrinsics=config.camera,
        start_index=start_index,
        max_candidates=max_candidates,
    )
