# data-pipeline

A thin driver layer over [Scene-Physics](../Scene-Physics) for **dataset
generation** and **model evaluation**. All the heavy lifting — object sampling,
XPBD settling, occlusion gating, USD/truth/priors export — lives in
`scene_physics.data_gen`; this repo only configures *which* scenes to generate,
*from which camera*, and *where* the outputs and evaluations go.

## Experiment-centric layout

Every experiment `<exp>` owns a parallel folder across four trees, plus one
config file:

```
config/<exp>.json     # the experiment's config — for now just the camera view
data_gen/<exp>/       # generation script(s) for the experiment
data/<exp>/           # generated dataset (gitignored output)
eval/<exp>/           # model evaluation for the experiment
src/data_pipeline/    # thin shared lib used by every experiment's scripts
```

Shared code (`src/data_pipeline/`):

- `paths.py` — resolves the per-experiment folders relative to the repo root, so
  a script writes to the right `data/<exp>/` regardless of the launch directory.
- `config.py` — loads `config/<exp>.json` and turns its embedded `camera` block
  into a `scene_physics` `CameraIntrinsics`.
- `runner.py` — `run_datagen(exp, spec, n_scenes, ...)` wires the experiment's
  camera and output dir into `generate_dataset_parallel`.

A `config/<exp>.json` embeds a copy of a Scene-Physics camera layout under a
`"camera"` key (the format is a superset of
`Scene-Physics/resources/camera_configs/*.json`). exp01 uses `p01_f03`
(eye-level front, 34° FOV, 640×480).

## Running a generation

```bash
# Full dataset for exp01 (camera p01_f03), 100 scenes:
NUM_WORLDS=64 uv run python data_gen/exp01/generate.py --n 100

# Quick end-to-end smoke test:
uv run python data_gen/exp01/generate.py --n 5 --num-worlds 5
```

Output lands in `data/exp01/` as `sceneNNN/` dirs plus `scene_stats.txt`.
The GPU batch size is the `NUM_WORLDS` env var (scene_physics default 5; ~64
suits an 80 GB H100, lower it on smaller cards); `--num-worlds` overrides it
explicitly. `--seed` makes runs reproducible; vary it to draw independent
additional scenes.

## Adding a new experiment

1. `config/<exp>.json` — embed the camera view under `"camera"`.
2. `data_gen/<exp>/generate.py` — define a `SceneSpec` and call
   `run_datagen("<exp>", spec, n_scenes=...)`.
3. `eval/<exp>/` — evaluation scripts/results for that dataset.
