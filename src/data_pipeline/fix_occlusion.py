"""Per-scene worker that builds one exp02 scene from its exp01 counterpart.

exp02 is exp01's dataset with each slightly-visible ``square_wood_block`` target
made fully hidden by sliding its occluder (see
``scene_physics.data_gen.occluder_fix.fix_scene``). This worker handles a single
scene end to end:

  * copy ``data/<src>/<scene>/data`` into ``data/exp02/<scene>``;
  * call ``fix_scene`` with the experiment camera (the view the scene was
    occlusion-gated against);
  * ``already``  -> the target is already hidden: carry the exp01 render
    artifacts over too (the scene is unchanged, so its renders are still valid);
  * ``hidden``   -> ``fix_scene`` rewrote the scene's data; leave ``results/``
    empty for a fresh Blender render;
  * ``dropped``  -> the occluder cannot hide the target: remove the scene dir.

It is launched as its own process, one per scene, by ``data_gen/exp02/generate.py``
— a fresh process per scene is the most complete GC there is (the OS reclaims
Warp's CUDA mempool on exit, so memory can't creep across scenes) and it
sandboxes the sweep. ``scene_physics`` is imported lazily here so the launcher
holds no CUDA context. The worker prints one ``FIXLINE`` the launcher parses::

    python -m data_pipeline.fix_occlusion exp02 scene002
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from data_pipeline.config import load_experiment_config
from data_pipeline.paths import data_dir

__all__ = ["SRC_EXP", "RENDER_ARTIFACTS"]

# exp02 derives from exp01: same scenes, occluder slid to fully hide the target.
SRC_EXP = "exp01"

# Render outputs that are valid to carry over for an unchanged ("already") scene
# (the scene-definition lives in data/; everything else in results/ is eval junk).
RENDER_ARTIFACTS = (
    "render.png",
    "seg_raw.png",
    "seg_labels.json",
    "segmentation.png",
    "segmentation_overlay.png",
    "segmentation_labels.json",
)


def main(exp: str, scene: str) -> None:
    from scene_physics.data_gen.occluder_fix import fix_scene

    src = data_dir(SRC_EXP) / scene
    out = data_dir(exp, create=True) / scene
    camera = load_experiment_config(exp).camera

    # Start from a clean copy of the source scene's data; results is (re)made by
    # fix_scene on a hidden outcome, else filled below for an unchanged scene.
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src / "data", out / "data")
    (out / "results").mkdir(parents=True, exist_ok=True)

    r = fix_scene(src, out, camera)

    if r.status == "dropped":
        shutil.rmtree(out)
    elif r.status == "already":
        for fn in RENDER_ARTIFACTS:
            f = src / "results" / fn
            if f.exists():
                shutil.copy2(f, out / "results" / fn)

    dx, dy = r.offset
    print(
        f"FIXLINE\t{scene}\t{r.status}\t{r.base_occ:.4f}\t{r.best_occ:.4f}\t"
        f"{r.occluder}\t{dx:.3f}\t{dy:.3f}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m data_pipeline.fix_occlusion <exp> <scene>")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])
