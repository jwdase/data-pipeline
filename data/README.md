# data/

Generated datasets, one subdirectory per experiment (`data/<exp>/`).

These are **build artifacts** and are gitignored — regenerate them with the
experiment's generation script, e.g.:

```bash
uv run python data_gen/exp01/generate.py --n 100 --num-worlds 64
```

Each run writes `sceneNNN/` dirs (with `data/` USD + truth/priors/makeup JSON and
a `results/` dir) plus a `scene_stats.txt` summary.
