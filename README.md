# data-pipeline

A thin driver layer over [Scene-Physics](../Scene-Physics) for **dataset
generation** and **model evaluation**. All the heavy lifting — object sampling,
XPBD settling, occlusion gating, USD/truth/priors export, rendering, and
importance-sampling inference — lives in `scene_physics.*`; this repo only
configures *which* scenes to generate, *from which camera*, and *where* the
outputs and evaluations go.

## Requirements

- **Python 3.12 + [uv](https://docs.astral.sh/uv/).** Run everything through
  `uv run …` so the editable `scene-physics` (sibling checkout at
  `../Scene-Physics`) and Warp deps resolve from `uv.lock`.
- **Blender 5.1.2** for rendering — point `$BLENDER` at the binary. Blender 4.0.x
  won't work (it can't import USD). Rendering uses a CUDA GPU.
- **`NUM_WORLDS`** env var = the GPU batch size for generation/inference
  (Scene-Physics default 5). Raise it on big GPUs (e.g. `NUM_WORLDS=64` for an
  80 GB H100) and lower it on small-VRAM cards to avoid OOM.

## Experiment-centric layout

Every experiment `<exp>` owns a parallel folder across four trees, plus one
config file:

```
config/<exp>.json     # the experiment's config — for now just the camera view
data_gen/<exp>/       # generation script(s) for the experiment (see "Bulk generation")
data/<exp>/           # generated dataset (gitignored output)
eval/<exp>/           # model evaluation for the experiment
src/data_pipeline/    # thin shared lib used by every experiment's scripts
```

`config/<exp>.json` embeds a copy of a Scene-Physics camera layout under a
`"camera"` key (a superset of `../Scene-Physics/resources/camera_configs/*.json`).
The shipped `data-set` / `exp01` configs use `p01_f03` (eye-level front, 34° FOV,
640×480).

Per scene, outputs follow a strict layout whose filenames Scene-Physics reads
verbatim:

```
data/<exp>/<scene>/
  data/      <scene>_physics.usdc  <scene>_truth.json  <scene>_makeup.json  <scene>_priors.json
  results/   render.png  segmentation*.png  target.json  seg_labels.json
```

### Shared lib (`src/data_pipeline/`)

- `utils/paths.py` — resolves the per-experiment folders relative to the repo
  root, so a script writes to the right `data/<exp>/` regardless of the launch
  directory (override with `$DATA_PIPELINE_ROOT`).
- `utils/config.py` — loads `config/<exp>.json` and turns its embedded `camera`
  block into a Scene-Physics `CameraIntrinsics`.
- `bulk_scene.py` — `run_datagen(exp, spec, n_scenes, ...)` wires the
  experiment's camera and output dir into `generate_dataset_parallel`.
- `gen/` — author a physics USD plus truth/priors/makeup JSON (`gen_physics`,
  `gen_truth`, `gen_json`).
- `render/` — Blender render plus occluded-target fill (`render`,
  `render_target`).

## Per-scene pipeline (manually authored scenes)

Many scenes we want to run inverse graphics on can't be made by program
synthesis — randomly dropping objects won't create the physical relationships we
want. For those, author the scene by hand and feed it through the pipeline:

```
Blender → .usdc export (Z-axis up) → gen_scene
```

> **A raw Blender export is _not_ a physics USD.** It carries no
> UsdPhysics/RigidBodyAPI schemas, so the simulator imports geometry but nothing
> moves and segmentation renders blank. Always promote it through `gen_scene` /
> `resolve_scene` (which call `to_physics`) — never copy a raw export straight
> into the `<scene>_physics.usdc` slot.

### Staging a new scene (`gen_scene`)

Drop the Blender export at `stage/<origin>.usdc`, then promote it into the scene
layout — this adds physics schemas, builds truth, renders, and writes
`target` / `priors` / `makeup`:

```bash
uv run src/data_pipeline/gen_scene.py --origin scene001 --exp data-set --scene scene052 --device GPU
```

### Resolving a raw export (`resolve_scene`)

If a raw, non-physics export was dropped straight into a scene's `data/` (at
`data/data-set/<scene>/data/<scene>.usdc`), `resolve_scene` runs the fix-up —
`to_physics` → `gen_truth` → render → verify — in one command:

```bash
uv run eval/data-set/resolve_scene.py scene033 scene034     # convert + truth + render
uv run eval/data-set/resolve_scene.py scene033 --no-render  # data only (fast, no GPU)
```

### Rendering (`gen_png`)

Render scenes to a PNG + segmentation mask from the experiment's camera:

```bash
uv run eval/data-set/gen_png.py                              # all data-set scenes
uv run eval/data-set/gen_png.py --scenes scene006 scene007  # a subset
uv run eval/data-set/gen_png.py --device CPU --samples 64    # quick check
```

### Watching a scene settle in Newton (`simulation`)

Re-simulate a scene's physics USD with Scene-Physics' XPBD solver — interactive
viewer, or record the sim to a USD:

```bash
uv run python -m data_pipeline.simulation --exp data-set --scene scene001
uv run python -m data_pipeline.simulation --exp data-set --scene scene001 --output sim.usdc
```

### Sim-sampling / inference (`inference`)

Run physics-based importance sampling to infer the hidden object's placement
(one process per scene):

```bash
uv run python -m data_pipeline.inference --exp data-set --scene scene001
uv run python -m data_pipeline.inference --exp data-set --scene scene001 --iterations 100
```

`--iterations` (Gibbs iterations) defaults to `$NUM_EPOCHS` (else 50).

## Bulk scene generation (programmatic)

For procedurally generated datasets, `run_datagen` batches Scene-Physics'
oversample-and-filter generator. Add a `data_gen/<exp>/generate.py` that
describes the scene (a `SceneSpec`) and calls it:

```python
from scene_physics.data_gen import SceneSpec
from data_pipeline import run_datagen

spec = SceneSpec(target=..., pool=..., n_mid=2, n_small=3)
run_datagen("exp01", spec, n_scenes=100)
```

```bash
NUM_WORLDS=64 uv run python data_gen/exp01/generate.py --n 100   # full run
uv run python data_gen/exp01/generate.py --n 5 --num-worlds 5    # smoke test
```

> The `data_gen/` tree isn't populated yet — `run_datagen` is ready in the lib,
> but the per-experiment `generate.py` scripts still need to be written. The
> pipeline exercised today is the manual per-scene flow above, on the `data-set`
> experiment.

## Adding a new experiment

1. `config/<exp>.json` — embed the camera view under `"camera"`.
2. `data_gen/<exp>/generate.py` — define a `SceneSpec` and call
   `run_datagen("<exp>", spec, n_scenes=...)`.
3. `eval/<exp>/` — evaluation scripts/results for that dataset.
