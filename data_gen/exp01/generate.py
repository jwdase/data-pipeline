"""Generate the exp01 Drop-&-Settle dataset.

Experiment exp01 renders scenes from camera ``p01_f03`` (embedded in
``config/exp01.json``). The scene recipe below mirrors the canonical
Scene-Physics benchmark: a low ``square_wood_block`` target laid flat, occluded
from the camera by a height-weighted mid-height object, with several surrounders
(one usually rested on the target as the physics-interaction probe).

Run from anywhere in the repo (the camera and output dir are resolved by
experiment id, not cwd). The GPU batch size comes from the ``NUM_WORLDS`` env
var unless ``--num-worlds`` is given::

    NUM_WORLDS=64 uv run python data_gen/exp01/generate.py --n 100
    uv run python data_gen/exp01/generate.py --n 5 --num-worlds 5   # quick smoke test

Output lands in ``data/exp01/`` as ``sceneNNN/`` dirs plus ``scene_stats.txt``.
"""

from __future__ import annotations

import argparse

from scene_physics.data_gen import SceneSpec

from data_pipeline import run_datagen

# Scene recipe (see Scene-Physics data_gen/scene_gen.py __main__ for rationale).
# "mid_height" objects (~13-27 cm) are the occluder candidates for the low block;
# "small" objects are the surrounders, several squat enough to be stacked on it.
POOL = {
    "mid_height": [
        "jug04", "bee", "glass1", "int_kitchen_accessories_le_creuset_bowl_30cm",
        "b05_coffee_grinder",        # ~18 cm (rescaled from native 1.17 m)
        "b04_candle_holder_metal",   # ~22 cm
        "vase_05",                   # ~27 cm
    ],
    "small": [
        "b03_loafbread", "bung", "pepper", "coffeemug",
        "shark", "heart", "banana_fix2", "star_wood_block",
        "round_coaster_stone", "f10_apple_iphone_4",
    ],
}

SPEC = SceneSpec(target="square_wood_block", pool=POOL, n_mid=2, n_small=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the exp01 dataset.")
    parser.add_argument("--n", "--n-scenes", dest="n_scenes", type=int, default=100,
                        help="Number of validated scenes to keep (default: 100).")
    parser.add_argument("--num-worlds", type=int, default=None,
                        help="Worlds settled per GPU batch. Omit to use the NUM_WORLDS env var "
                             "(scene_physics default 5); ~64 suits an 80 GB H100.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed; vary for independent additional scenes (default: 0).")
    parser.add_argument("--start-index", type=int, default=1,
                        help="First sceneNNN number, to extend a dir without colliding (default: 1).")
    args = parser.parse_args()

    run_datagen(
        "exp01",
        SPEC,
        n_scenes=args.n_scenes,
        num_worlds=args.num_worlds,
        seed=args.seed,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
