"""End-to-end "resolve color" pipeline for an exp05 scene.

Packages the by-hand sequence we ran to fix a raw Blender export dropped into
exp05 into one command, per scene:

  1. to_physics : raw export (no UsdPhysics schemas; cameras/lights; meshes nested
                  /root/root/<base>/<base>_NNN) -> proper physics USD
                  (/root/<name>/<name> + RigidBody, cameras/lights dropped). Idempotent:
                  skipped when the scene's *_physics.usdc already carries rigid bodies.
  2. gen_truth  : (re)author <scene>_truth.json from the physics USD's prim names+poses,
                  so the render's segmentation selector matches every object's name.
  3. render     : render.png + segmentation (one Blender subprocess per scene). Object
                  materials resolve automatically -- material_specs.spec_for now maps
                  _NNN instance suffixes to their base spec, so nothing renders flat gray.
  4. verify     : rigid bodies present, truth keys == USD mesh names, segmentation
                  labels every object (not a blank/white mask).

INPUT CONTRACT: drop the raw, non-physics export at

    data/exp05/<scene>/data/<scene>.usdc

then run::

    uv run eval/exp05/resolve_scene.py scene033 scene034     # convert + truth + render
    uv run eval/exp05/resolve_scene.py scene033 --no-render  # data only (fast, no GPU)

A raw export mistakenly saved into the *_physics.usdc slot is still handled: it is
backed up to <scene>_physics.usdc.rawbak, promoted to <scene>.usdc, and converted.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from pxr import Usd, UsdPhysics

# Run as a script: this file's dir is sys.path[0], so the sibling eval scripts import
# by bare name. The heavy work in each lives under its own __main__ guard, so importing
# them here only pulls in the functions (and their scene_physics deps).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_physics import convert as to_physics_convert  # noqa: E402
from gen_truth import build_truth  # noqa: E402

from data_pipeline.render import run_render  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "data" / "exp05"


def _has_rigid_bodies(usd_path: Path) -> bool:
    """True if the USD authors at least one dynamic body (vs. a raw mesh export)."""
    stage = Usd.Stage.Open(str(usd_path))
    return any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())


def _to_physics(scene: str) -> Path:
    """Step 1: ensure <scene>_physics.usdc is a real physics USD; return its path."""
    data = EXP_DIR / scene / "data"
    phys = data / f"{scene}_physics.usdc"
    raw = data / f"{scene}.usdc"

    if not raw.exists():
        if phys.exists() and _has_rigid_bodies(phys):
            print(f"[{scene}] 1/4 to_physics: {phys.name} already valid -- skip")
            return phys
        if phys.exists():  # raw export dropped into the _physics slot -> recover it
            shutil.copy2(phys, data / f"{scene}_physics.usdc.rawbak")
            shutil.copy2(phys, raw)
            print(f"[{scene}] 1/4 to_physics: {phys.name} was a raw export "
                  f"(backed up .rawbak, promoted to {raw.name})")
        else:
            raise SystemExit(
                f"[{scene}] no input USD -- drop the raw export at {raw}"
            )

    out = to_physics_convert(raw, out_dir=EXP_DIR)
    if not _has_rigid_bodies(out):
        raise SystemExit(f"[{scene}] to_physics authored no rigid bodies from {raw.name}")
    return out


def _gen_truth(scene: str, phys: Path) -> None:
    """Step 2: rewrite <scene>_truth.json from the physics USD's prim names + poses."""
    out = phys.with_name(f"{scene}_truth.json")
    truth = build_truth(phys)
    out.write_text(json.dumps(truth, indent=4))
    print(f"[{scene}] 2/4 gen_truth: {out.name} ({len(truth)} objects)")


def _verify(scene: str, rendered: bool) -> bool:
    """Step 4: physics + truth + segmentation are mutually consistent."""
    data = EXP_DIR / scene / "data"
    stage = Usd.Stage.Open(str(data / f"{scene}_physics.usdc"))
    mesh = sorted(p.GetName() for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    rb = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    truth = sorted(json.loads((data / f"{scene}_truth.json").read_text()))

    problems = []
    if rb == 0:
        problems.append("no rigid bodies")
    if truth != mesh:
        problems.append(f"truth keys != USD meshes (truth {len(truth)} vs mesh {len(mesh)})")
    if rendered:
        labels = json.loads((EXP_DIR / scene / "results" / "segmentation_labels.json").read_text())
        labelled = [k for k in labels if k != "background"]
        if sorted(labelled) != mesh:
            problems.append(f"segmentation labels {len(labelled)} != {len(mesh)} objects (blank mask?)")

    status = "OK" if not problems else "FAIL: " + "; ".join(problems)
    print(f"[{scene}] 4/4 verify: {status}")
    return not problems


def resolve(scene: str, *, render: bool, samples: int, device: str) -> bool:
    phys = _to_physics(scene)
    _gen_truth(scene, phys)
    if render:
        print(f"[{scene}] 3/4 render: rendering ...")
        run_render("exp05", [scene], samples=samples, device=device)
    else:
        print(f"[{scene}] 3/4 render: skipped (--no-render)")
    return _verify(scene, render)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert + truth + render an exp05 scene from its raw USD.")
    ap.add_argument("scenes", nargs="+", help="scene names, e.g. scene033 (raw input at data/exp05/<scene>/data/<scene>.usdc)")
    ap.add_argument("--no-render", action="store_true", help="run to_physics + gen_truth only (no GPU/Blender)")
    ap.add_argument("--samples", type=int, default=128, help="Cycles samples (default: 128)")
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"], help="render device (default: GPU)")
    args = ap.parse_args()

    results = {s: resolve(s, render=not args.no_render, samples=args.samples, device=args.device)
               for s in args.scenes}

    ok = [s for s, good in results.items() if good]
    bad = [s for s, good in results.items() if not good]
    print(f"\nresolved {len(ok)}/{len(results)}: ok={ok or '-'}  failed={bad or '-'}")
    sys.exit(1 if bad else 0)
