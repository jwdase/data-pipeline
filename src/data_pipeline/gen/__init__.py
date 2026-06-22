"""Scene-definition generation: author the per-scene data a render/sim needs.

``gen_physics`` re-authors a raw USD into a physics-enabled ``*_physics.usdc``;
``gen_truth`` writes the ``*_truth.json`` poses; ``gen_json`` writes the
``*_makeup.json`` / ``*_priors.json`` the model pipeline consumes.
"""

from data_pipeline.gen.gen_physics import convert
from data_pipeline.gen.gen_truth import build_truth, build_save_truth
from data_pipeline.gen.gen_json import create_prior_makeup

__all__ = ["convert", "build_truth", "build_save_truth", "create_prior_makeup"]
