"""Render runner — the thin bridge from an experiment to Scene-Physics' renderer.

The mirror image of :func:`data_pipeline.runner.run_datagen`: an ``eval/<exp>/``
script only says *which* scenes to render; this module resolves the experiment's
camera from ``config/<exp>.json``, finds the generated scenes under
``data/<exp>/``, and calls Scene-Physics'
``visualization.render_pipeline.render_scene`` for each — so every scene is
rendered to a PNG + segmentation mask from the exact camera view the dataset was
generated and occlusion-gated against.

For each scene ``render_scene`` writes, into ``<scene>/results/`` (or
``<out_root>/<scene>/``):

    render.png                 -- photoreal beauty render
    seg_raw.png                -- flat per-object ID render (Blender)
    segmentation.png           -- single-channel mask, pixel value == object id
    segmentation_overlay.png   -- mask alpha-blended over render.png
    segmentation_labels.json   -- {name: id} (0 = background)

Blender is required (on PATH, or point ``$BLENDER`` at the binary). The heavy
lifting — geometry, lighting, the flat-ID decode — all lives in Scene-Physics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from scene_physics.visualization.render_pipeline import DEFAULT_HDRI, render_scene

from data_pipeline.config import load_experiment_config
from data_pipeline.paths import data_dir

__all__ = ["scene_dirs", "run_render"]


def scene_dirs(exp: str, scenes: list[str] | None = None) -> list[Path]:
    """Scene dirs to render for ``exp``: the named ones, else every
    ``sceneNNN/`` in ``data/<exp>/``. A dir is skipped (with a warning) unless
    it holds a ``data/*_physics.usdc`` — the USD ``render_scene`` imports."""
    root = data_dir(exp)
    candidates = (
        [root / n for n in scenes]
        if scenes
        else sorted(p for p in root.glob("scene*") if p.is_dir())
    )

    dirs: list[Path] = []
    for d in candidates:
        if next(d.glob("data/*_physics.usdc"), None) is None:
            print(f"[run_render] skip {d.name}: no data/*_physics.usdc")
            continue
        dirs.append(d)
    return dirs


def run_render(
    exp: str,
    scenes: list[str] | None = None,
    *,
    samples: int = 128,
    device: str = "GPU",
    scale: float = 1.0,
    hdri: str | Path | None = None,
    view_transform: str = "AgX",
    out_root: str | Path | None = None,
) -> list[Path]:
    """Render scenes for experiment ``exp`` to PNG + segmentation mask.

    The viewpoint comes from ``config/<exp>.json`` (the experiment's camera);
    everything else is passed straight through to ``render_scene``. ``scenes``
    limits the run to specific ``sceneNNN`` names (default: all in
    ``data/<exp>/``). ``out_root`` redirects per-scene output to
    ``<out_root>/<scene>/`` instead of the default ``<scene>/results/``.

    Fidelity levers: ``samples`` (Cycles samples — less noise) and ``scale``, a
    resolution multiplier applied to the camera's width/height. ``scale`` keeps
    the canonical framing exactly (vertical FOV is fixed and 640x480 is 4:3, so
    scaling both axes only supersamples) while sharpening both the render and the
    mask. Use ``device="GPU"`` to afford higher settings.

    Returns the list of results dirs written (one per rendered scene).
    """
    camera = load_experiment_config(exp).camera
    if scale != 1.0:
        # Supersample the canonical view without touching framing: scale both
        # axes (preserves 4:3 aspect) and leave fov/eye/target/up untouched.
        camera = replace(
            camera,
            width=round(camera.width * scale),
            height=round(camera.height * scale),
        )
    hdri = DEFAULT_HDRI if hdri is None else hdri

    dirs = scene_dirs(exp, scenes)
    if not dirs:
        raise FileNotFoundError(
            f"No scenes to render in {data_dir(exp)} "
            f"(pass scenes=['scene001', ...] or generate the dataset first)."
        )

    print(
        f"[run_render] exp={exp} scenes={len(dirs)} samples={samples} device={device}\n"
        f"             camera fov={camera.fov_degree}deg "
        f"{camera.width}x{camera.height} (scale={scale}) eye={camera.eye.tolist()}"
    )

    results: list[Path] = []
    for d in dirs:
        out_dir = Path(out_root) / d.name if out_root is not None else None
        print(f"[run_render] rendering {d.name} ...")
        out = render_scene(
            d,
            intr=camera,
            hdri=hdri,
            samples=samples,
            device=device,
            view_transform=view_transform,
            out_dir=out_dir,
        )
        print(f"[run_render] {d.name} -> {out}")
        results.append(out)
    return results
