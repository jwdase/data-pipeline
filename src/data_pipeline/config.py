"""Experiment configuration.

Each experiment has one config file at ``config/<exp>.json``. The config is a
small JSON object whose only meaningful payload, for now, is the camera view
under a ``"camera"`` key (an embedded copy of a Scene-Physics camera layout,
e.g. ``resources/camera_configs/p01_f03.json``). Wrapping the camera under a key
rather than making the file *be* the camera leaves room for the config to grow
(target/pool/scene counts) without changing the format.

``load_experiment_config`` parses it into an ``ExperimentConfig`` whose
``camera`` is a ready-to-use ``scene_physics`` ``CameraIntrinsics`` — the same
object ``data_gen.scene_gen`` takes as its ``intrinsics`` argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scene_physics.configs.camera import CameraIntrinsics, camera_from_dict

from data_pipeline.paths import config_path

__all__ = ["ExperimentConfig", "load_experiment_config"]


@dataclass(frozen=True)
class ExperimentConfig:
    """A parsed experiment config.

    ``name`` is the experiment id (folder name). ``camera`` is the rendering
    viewpoint/intrinsics used by the datagen occlusion checks and downstream
    eval. ``raw`` keeps the full parsed JSON so future fields are reachable
    without a code change here.
    """

    name: str
    camera: CameraIntrinsics
    raw: dict


def load_experiment_config(exp: str) -> ExperimentConfig:
    """Load ``config/<exp>.json`` into an :class:`ExperimentConfig`."""
    path: Path = config_path(exp)
    if not path.exists():
        raise FileNotFoundError(
            f"No config for experiment {exp!r} at {path}. "
            f"Create it (an embedded camera view under a 'camera' key)."
        )
    raw = json.loads(path.read_text())
    if "camera" not in raw:
        raise KeyError(f"{path}: config must contain a 'camera' key (the camera view).")
    return ExperimentConfig(name=exp, camera=camera_from_dict(raw["camera"]), raw=raw)
