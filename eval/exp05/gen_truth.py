"""Author the missing <Scene>_truth.json beside each exp05 physics USD.

The exp05 scenes are hand-authored (manual/ -> to_physics.py) and ship with only a
*_physics.usdc -- no *_truth.json. data_pipeline.run_render needs one: render_scene calls
scene_physics...render_pipeline.scene_names, which reads the keys of data/*_truth.json to
learn which prims to segment. Without it the render aborts with StopIteration.

This reads the same physics USD render imports, takes every object Xform under /root
(skipping PhysicsScene + the physics Materials), and writes name -> world pose
[x, y, z, qx, qy, qz, qw] -- the schema scene_gen's truth.json uses (and the same
T+R decomposition as eval/exp05/to_physics.py). Because the names come straight from the
USD render loads, the truth keys line up exactly with the prims Blender segments.

Idempotent and backwards compatible: existing experiments already have their truth.json,
and this only writes the file when it's absent (use --force to overwrite).

    uv run python eval/exp05/gen_truth.py            # all exp05 scenes missing truth.json
    uv run python eval/exp05/gen_truth.py --force    # rewrite every truth.json
"""

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom, Gf

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "data" / "exp05"

# /root children that are scene scaffolding, not segmentable objects.
_SKIP_TYPES = {"PhysicsScene", "Material", "Scope"}


def _world_pose(prim, xform_cache):
    """World matrix -> [x, y, z, qx, qy, qz, qw] (translation, then quat imag xyz + real w)."""
    xf = Gf.Transform(xform_cache.GetLocalToWorldTransform(prim))
    t = xf.GetTranslation()
    q = xf.GetRotation().GetQuat()
    im = q.GetImaginary()
    return [t[0], t[1], t[2], im[0], im[1], im[2], q.GetReal()]


def build_truth(usd_path):
    """{object name: world pose} for the object Xforms directly under /root."""
    stage = Usd.Stage.Open(str(usd_path))
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    truth = {}
    for prim in stage.Traverse():
        if prim.GetParent().GetName() != "root" or prim.GetTypeName() in _SKIP_TYPES:
            continue
        truth[prim.GetName()] = _world_pose(prim, xform_cache)
    return truth


def main():
    ap = argparse.ArgumentParser(description="Write <Scene>_truth.json for exp05 scenes.")
    ap.add_argument("--force", action="store_true", help="overwrite existing truth.json")
    args = ap.parse_args()

    usds = sorted(EXP_DIR.glob("scene*/data/*_physics.usdc"))
    if not usds:
        ap.error(f"no */data/*_physics.usdc under {EXP_DIR}")

    for usd in usds:
        out = usd.with_name(usd.name.replace("_physics.usdc", "_truth.json"))
        # if out.exists() and not args.force:
        #    print(f"[skip] {out.relative_to(ROOT)} (exists)")
        #    continue
        truth = build_truth(usd)
        out.write_text(json.dumps(truth, indent=4))
        print(f"[done] {out.relative_to(ROOT)} ({len(truth)} objects: {', '.join(truth)})")


if __name__ == "__main__":
    main()
