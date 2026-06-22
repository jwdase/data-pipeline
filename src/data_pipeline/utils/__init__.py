"""Shared infrastructure: experiment config + project path conventions.

``config`` resolves ``config/<exp>.json`` into a ready-to-use camera; ``paths``
defines the per-experiment folder layout everything else reads and writes.
"""

from data_pipeline.utils import paths
from data_pipeline.utils.config import ExperimentConfig, load_experiment_config
from data_pipeline.utils.paths import config_path, data_dir, project_root

__all__ = [
    "paths",
    "ExperimentConfig",
    "load_experiment_config",
    "config_path",
    "data_dir",
    "project_root",
]
