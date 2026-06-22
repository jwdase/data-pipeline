"""Render each data-set scene's occluded target to a flat-color ``target.png``.

A thin CLI over :func:`data_pipeline.run_render_targets`. For every scene that has
already been rendered (``results/segmentation.png`` present, e.g. via
``gen_png.py``), this finds the occluded target — the object that all but vanishes
from the segmentation — and writes its full outline filled with the exact color the
scene was generated with, plus a ``target.json`` recording the visible fraction (the
"little error" for a not-quite-hidden target).

The viewpoint is exp01's camera ``p01_f03`` from ``config/data-set.json`` — the same
view ``gen_png.py`` rendered from — reproduced on the CPU, so no Blender/GPU is
needed. Run from anywhere in the repo::

    uv run python eval/data-set/gen_target_png.py                          # all scenes
    uv run python eval/data-set/gen_target_png.py --scenes scene015        # a subset

Outputs land in ``data/data-set/sceneNNN/results/`` (target.png, target.json) unless
``--out-root`` redirects them.
"""

from __future__ import annotations

import argparse

from data_pipeline import run_render_targets


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render data-set scenes' occluded targets to flat-color PNGs "
        "(camera p01_f03, color from MATERIAL_SPECS)."
    )
    ap.add_argument(
        "--scenes", nargs="*", default=None,
        help="Scene names to render, e.g. scene015 (default: all scenes).",
    )
    ap.add_argument(
        "--out-root", default=None,
        help="Write target.png/target.json to <out-root>/<scene>/ instead of the "
        "default <scene>/results/.",
    )
    args = ap.parse_args()

    run_render_targets("data-set", scenes=args.scenes, out_root=args.out_root)


if __name__ == "__main__":
    main()
