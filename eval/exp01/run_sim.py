"""Run the exp01 model pipeline (physics-based importance sampling) per scene.

The model-side counterpart of ``gen_png.py``: instead of rendering, it runs
Scene-Physics' importance sampler over every ``data/exp01/`` scene to infer the
hidden object's placement, writing each scene's outputs into its ``results/`` dir.

Each scene runs as its own subprocess (``python -m data_pipeline.inference``) —
the same pattern as ``view.py``. A fresh process per scene is the most complete
garbage collection there is (the OS reclaims Warp's CUDA mempool and JAX's pool
when the child exits, so nothing creeps across scenes) and it sandboxes the
sweep, so one scene crashing doesn't abort the rest. This launcher itself never
imports ``scene_physics``, so it holds no CUDA context.

Run from anywhere in the repo; scenes resolve by path, not cwd::

    uv run python eval/exp01/run_sim.py                    # every scene, one at a time
    uv run python eval/exp01/run_sim.py scene006 scene007  # just these
    NUM_WORLDS=64 NUM_EPOCHS=50 uv run python eval/exp01/run_sim.py   # tune the sampler

Set ``JOBS`` to run several scenes at once, all sharing the GPU (the driver
time-slices them). VRAM is the cap — on an 8 GiB card expect ~2-3 before OOM;
watch ``nvidia-smi`` and lower ``NUM_WORLDS`` to fit more::

    JOBS=2 uv run python eval/exp01/run_sim.py
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = "exp01"
DATA = Path(__file__).resolve().parents[2] / "data" / EXP
WORKER = "data_pipeline.inference"   # python -m data_pipeline.inference <exp> <scene>
JOBS = int(os.environ.get("JOBS", 1))   # scenes to run concurrently, sharing the GPU


def scene_names(names):
    """The scene names to run: the given ones, else every sceneNNN in
    data/exp01/. Skip (with a warning) any lacking a settled physics USD."""
    names = names or sorted(p.name for p in DATA.glob("scene*") if p.is_dir())
    for name in names:
        if (DATA / name / "data" / f"{name}_physics.usdc").exists():
            yield name
        else:
            print(f"[run_sim] skip {name}: no {name}_physics.usdc")


def run_one(name):
    """Launch one scene as a subprocess and report its exit status."""
    print(f"[run_sim] {name} ...")
    rc = subprocess.run([sys.executable, "-m", WORKER, EXP, name]).returncode
    print(f"[run_sim] {name}: " + ("done" if rc == 0 else f"FAILED (exit {rc})"))


if __name__ == "__main__":
    names = list(scene_names(sys.argv[1:]))
    # JOBS threads, each supervising one subprocess at a time; JOBS=1 is sequential.
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        list(pool.map(run_one, names))
