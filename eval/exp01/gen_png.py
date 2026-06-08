"""Render every exp01 scene to a PNG + segmentation mask from camera ``p01_f03``.

A thin CLI over :func:`data_pipeline.run_render`. The viewpoint is loaded from
``config/exp01.json`` (camera ``p01_f03``, eye-level front, 34 deg FOV) — the
same camera the dataset was generated and occlusion-gated against — so each
scene is rendered from the exact perspective the experiment is defined by.

Blender is required (on PATH, or point ``$BLENDER`` at the binary). Run from
anywhere in the repo — scenes and the camera resolve by experiment id, not cwd::

    uv run python eval/exp01/gen_png.py                 # all scenes in data/exp01/
    uv run python eval/exp01/gen_png.py --scenes scene006 scene007   # a subset
    uv run python eval/exp01/gen_png.py --device CPU --samples 64     # quick check

Outputs land in ``data/exp01/sceneNNN/results/`` (render.png, segmentation.png,
segmentation_overlay.png, ...) unless ``--out-root`` redirects them.
"""

from __future__ import annotations

import argparse

from data_pipeline import run_render

EXP = "exp01"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render exp01 scenes to PNG + segmentation from camera p01_f03."
    )
    ap.add_argument(
        "--scenes", nargs="*", default=None,
        help="Scene names to render, e.g. scene006 (default: all scenes in data/exp01/).",
    )
    ap.add_argument("--samples", type=int, default=128, help="Cycles samples (default: 128).")
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"],
                    help="Blender render device (default: GPU).")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Resolution multiplier on the camera's 640x480 (default: 1.0). "
                         "e.g. 2 -> 1280x960; keeps the p01_f03 framing exactly.")
    ap.add_argument("--hdri", default=None,
                    help="HDRI environment map (default: Scene-Physics' lythwood_room_4k.hdr).")
    ap.add_argument("--view-transform", default="AgX",
                    help="Blender view transform (default: AgX).")
    ap.add_argument(
        "--out-root", default=None,
        help="Write results to <out-root>/<scene>/ instead of the default <scene>/results/.",
    )
    args = ap.parse_args()

    run_render(
        EXP,
        scenes=args.scenes,
        samples=args.samples,
        device=args.device,
        scale=args.scale,
        hdri=args.hdri,
        view_transform=args.view_transform,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
