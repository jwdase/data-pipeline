"""
Using scene-physics relies on a specific layout of files. The
purpose of this code is to enable the movement of a .usdc file in
the STAGED directory to become data that we can run the inference
model on. Doing such will allow us to use any scene generated
"""

from pathlib import Path
import argparse

from data_pipeline.gen.gen_physics import convert
from data_pipeline.render.render import run_render
from data_pipeline.gen.gen_truth import build_save_truth
from data_pipeline.render.render_target import run_render_targets
from data_pipeline.gen.gen_json import create_prior_makeup 

ORIGIN_BASE="stage"
ROOT = Path(__file__).resolve().parents[2]

def create_folders(exp, scene_number):
    data = ROOT / "data" / exp / scene_number / "data"
    results = ROOT / "data" / exp / scene_number / "results"
    
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)


def main(origin, exp, scene_number, device):
    # Step 0: Make the Folders
    create_folders(exp, scene_number)

    # Step 1: Copy USDC - ADD Physics
    initial_usdc_loc = (ROOT / ORIGIN_BASE / origin).with_suffix(".usdc")
    new_usdc_loc = (
        ROOT / "data" / exp / scene_number / "data" / f"{scene_number}_physics.usdc"
        )
    convert(initial_usdc_loc, new_usdc_loc)

    # Step 2: Build Truths
    build_save_truth(new_usdc_loc)

    # Step 2: Run render parse on it
    run_render(
        exp, 
        scenes=[scene_number], 
        samples=128,
        device=device,
        scale=1.0,
        hdri=None,
        view_transform="AgX",
        out_root=None
        )



    # Step 3: Create target.json
    run_render_targets(exp, scenes=[scene_number])


    # Step 4: Create prior.json, Create 
    root = ROOT / "data" / exp
    create_prior_makeup(root, scene_number)

    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Turn a .usdc into a complete testing scene"
    )

    ap.add_argument(
        "--origin", help="Specify the scene you would like converted"
    )
    ap.add_argument(
        "--exp", help="Specify the output experiment run you want file in"
    )
    ap.add_argument(
        "--scene", help="Specify the target scene you want this to be called?"
    )
    ap.add_argument(
        "--device", default="GPU", help="Specify the device to run this on GPU vs CPU",
        choices=["CPU", "GPU"]
    )

    args = ap.parse_args()

    main(args.origin, args.exp, args.scene, args.device)