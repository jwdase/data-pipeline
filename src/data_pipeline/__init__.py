"""data-pipeline — a thin driver layer over Scene-Physics for dataset
generation and model evaluation.

The repo is experiment-centric: see :mod:`data_pipeline.utils.paths` for the
per-experiment folder layout. Typical use from a ``data_gen/<exp>/`` script::

    from scene_physics.data_gen import SceneSpec
    from data_pipeline import run_datagen

    spec = SceneSpec(target=..., pool=..., n_mid=2, n_small=3)
    run_datagen("exp01", spec, n_scenes=100)
"""

from data_pipeline.utils.config import ExperimentConfig, load_experiment_config
from data_pipeline.bulk_scene import run_datagen
from data_pipeline.render import run_render, run_render_targets
from data_pipeline.utils import paths

# inference / simulation are per-scene CLI workers (``python -m
# data_pipeline.inference``); they are intentionally NOT imported here, so the
# package init doesn't pull them into sys.modules ahead of ``-m`` execution
# (which would trigger a runpy double-import RuntimeWarning). Import their
# functions directly: ``from data_pipeline.inference import run_inference``.

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "run_datagen",
    "run_render",
    "run_render_targets",
    "paths",
]
