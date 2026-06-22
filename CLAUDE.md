# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A thin Python driver over the sibling package **Scene-Physics** (`../Scene-Physics`, installed editable) for physics-based 3D scene generation, Blender rendering, and inverse-graphics inference. The heavy lifting — object sampling, XPBD settling, USD/truth/priors export, rendering, importance-sampling inference — lives in `scene_physics.*`; code here only configures *which* scenes, *which* camera, and *where* outputs go.

`src/data_pipeline/` is the shared lib (`bulk_scene.py`, `gen_scene.py`, `simulation.py`, `inference.py`, plus `gen/`, `render/`, `utils/`). Per-experiment evaluation scripts live under `eval/<exp>/`.

## Environment

- Python 3.12, managed with **uv**. Run everything through `uv run …` so the editable Scene-Physics + Warp deps resolve from `uv.lock`.
- **Blender 5.1.2 is required**, reached via the already-exported `$BLENDER`. Never use `/usr/bin/blender` (4.0.2) — it lacks USD import. Rendering needs a CUDA GPU; the `NUM_WORLDS` env var sets the GPU batch size — lower it on small-VRAM cards to avoid OOM.

## Commands (what actually runs today)

The active dataset is `data-set` (the only tree under `data/`). Run from the repo root:

```bash
# Promote a staged Blender USD into the scene layout
uv run src/data_pipeline/gen_scene.py --origin scene001 --exp data-set --scene scene052 --device GPU

# Full per-scene pipeline on a raw Blender export: to_physics + gen_truth + render
uv run eval/data-set/resolve_scene.py scene033 [--no-render] [--device GPU] [--samples 128]

# Render every data-set scene to PNG + segmentation mask (camera p01_f03)
uv run eval/data-set/gen_png.py [--device GPU] [--samples 128]

# Re-simulate / run inference on one scene (NOTE: --exp/--scene flags, not positional)
uv run python -m data_pipeline.simulation --exp data-set --scene scene001
uv run python -m data_pipeline.inference  --exp data-set --scene scene001
```

The README describes a bulk `data_gen/<exp>/generate.py --n N` flow, but that `data_gen/` tree does **not** exist in the working copy yet (`bulk_scene.run_datagen()` is the library it would call). Don't run those README commands as-is.

## Scene layout (load-bearing)

Per scene: `data/<exp>/<scene>/` with two subfolders whose filenames Scene-Physics requires verbatim:
- `data/` — `<scene>_physics.usdc`, `<scene>_truth.json`, `<scene>_makeup.json`, `<scene>_priors.json`
- `results/` — `render.png`, `segmentation*.png`, `target.json`, `seg_labels.json`

Paths resolve relative to the repo root (`utils/paths.py`), so scripts work from any cwd.

## Gotchas

- **A raw Blender export is not a physics USD.** A plain mesh dropped into the `<scene>_physics.usdc` slot has no UsdPhysics / RigidBodyAPI schemas → simulation breaks and segmentation renders blank. Always author it through `gen_scene.py` / `resolve_scene.py` (which call `to_physics.convert()`) instead of copying the raw export.
- **`<scene>_truth.json` must exist before rendering** — it maps object names → world poses, and the render's segmentation selector reads those names and aborts without it.

## Working in this repo

- Use **feature branches + PRs**; don't commit directly to `main`.
- No unit tests. After a change, **smoke-test** the affected path before calling it done (e.g. `resolve_scene.py <scene> --no-render` for non-render edits, or a small render / `--num-worlds 5` run).
