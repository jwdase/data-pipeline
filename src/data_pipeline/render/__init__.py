"""Rendering: photoreal render + segmentation, and the occluded-target fill.

``render`` drives Scene-Physics' Blender renderer (render.png + segmentation);
``render_target`` is its CPU companion that fills the hidden target's silhouette.
"""

from data_pipeline.render.render import run_render, scene_dirs
from data_pipeline.render.render_target import run_render_targets

__all__ = ["run_render", "scene_dirs", "run_render_targets"]
