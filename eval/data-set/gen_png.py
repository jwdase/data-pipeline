"""Render every data-set scene to a PNG + segmentation mask from camera ``p01_f03``.

A thin CLI over :func:`data_pipeline.run_render`. The viewpoint is loaded from
``config/data-set.json``, which embeds exp01's camera ``p01_f03`` (eye-level
front, 34 deg FOV) — so scenes render from the same perspective as exp01/exp02.

Blender is required (on PATH, or point ``$BLENDER`` at the binary). Run from
anywhere in the repo — scenes and the camera resolve by experiment id, not cwd::

    uv run python eval/data-set/gen_png.py                              # all scenes
    uv run python eval/data-set/gen_png.py --scenes scene006 scene007   # a subset
    uv run python eval/data-set/gen_png.py --device CPU --samples 64     # quick check

Outputs land in ``data/data-set/sceneNNN/results/`` (render.png,
segmentation.png, segmentation_overlay.png, ...) unless ``--out-root`` redirects
them.
"""

from __future__ import annotations

import argparse

from data_pipeline import run_render


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render data-set scenes to PNG + segmentation from camera p01_f03 (exp01's camera)."
    )
    ap.add_argument(
        "--scenes", nargs="*", default=None,
        help="Scene names to render, e.g. scene006 (default: all scenes).",
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
        "data-set",
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
