"""Project path conventions.

This repo is experiment-centric: every experiment ``<exp>`` owns a parallel
folder across four top-level trees, plus one config file::

    config/<exp>.json     # the experiment's config (currently just the camera view)
    data_gen/<exp>/       # generation script(s) for the experiment
    data/<exp>/           # generated dataset (gitignored output)
    eval/<exp>/           # model evaluation for the experiment

All paths are resolved relative to the project root, so a generation script
writes to the same ``data/<exp>/`` no matter which directory it is launched
from. The root is the repo checkout (three parents up from this file, since it
lives at ``src/data_pipeline/paths.py``); set ``DATA_PIPELINE_ROOT`` to override
(e.g. when the package is installed non-editably).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "project_root",
    "config_path",
    "data_gen_dir",
    "data_dir",
    "eval_dir",
]


def project_root() -> Path:
    """The data-pipeline repo root (env ``DATA_PIPELINE_ROOT`` wins if set)."""
    env = os.environ.get("DATA_PIPELINE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def config_path(exp: str) -> Path:
    """``config/<exp>.json`` — the experiment's config file."""
    return project_root() / "config" / f"{exp}.json"


def data_gen_dir(exp: str) -> Path:
    """``data_gen/<exp>/`` — where the experiment's generation script(s) live."""
    return project_root() / "data_gen" / exp


def data_dir(exp: str, *, create: bool = False) -> Path:
    """``data/<exp>/`` — output root for the experiment's generated dataset."""
    d = project_root() / "data" / exp
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def eval_dir(exp: str, *, create: bool = False) -> Path:
    """``eval/<exp>/`` — where the experiment's model-evaluation artifacts live."""
    d = project_root() / "eval" / exp
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
