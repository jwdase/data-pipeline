import subprocess
import sys
from pathlib import Path

# Resolve relative to this file so it works regardless of cwd
DATA = Path(__file__).resolve().parents[2] / "data" / "exp01"

# scene_physics.simulation.simulation's __main__ takes `<scene_usd> <output>`.
# run_simulation ignores the second arg, so "-" is just a placeholder.
SIM = "scene_physics.simulation.simulation"


def gen_files():
    for num in range(1, 101):
        scene = f"scene{num:03d}"
        usd = DATA / scene / "data" / f"{scene}_physics.usdc"
        if usd.exists():          # not every scene 1..100 may exist
            yield str(usd)


if __name__ == "__main__":
    # Run each scene as its own process instead of calling run_simulation in
    # this one. Process exit is the most complete garbage collection there is:
    # the OS reclaims every byte (Warp's CUDA mempool included) when the child
    # dies, so memory can't creep across the 100 scenes — no in-process gc or
    # device sync needed. It also sandboxes the sweep: one scene crashing
    # doesn't abort the rest.
    for val in gen_files():
        subprocess.run([sys.executable, "-m", SIM, val, "-"], check=False)
