"""
Author physics-enabled USDs from the hand-authored Blender scenes in manual/.

Blender's USD export writes plain Mesh prims with no UsdPhysics schemas, so
newton.ModelBuilder.add_usd imports zero bodies/shapes and the sim/render comes up empty
(see eval/data-set/run_sim.py). This reads each mesh's local geometry + world pose out of a
manual USD and re-authors it through scene_physics.data_gen.usd_export.write_layout_usd --
the same function scene_gen uses for the dataset -- tagging the table as a static collider
and every other object as a dynamic rigid body.

Output is written to data/data-set/<scene>/data/<scene>_physics.usdc -- the nested, lowercase
per-scene layout the rest of data-set (run_sim/render/inference/gen_truth) discovers via
scene*/data/*_physics.usdc, not flat in data/data-set/. The "_physics.usdc" suffix is load-
bearing: run_simulation's table-collider/makeup logic keys off it, and with no makeup.json
sibling it falls back to scene_gen's default table, which is correct for these scenes.
"""

from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Gf

from scene_physics.data_gen.scene_gen import GRAVITY, MU, RESTITUTION, TABLE
from scene_physics.data_gen.usd_export import UsdBody, write_layout_usd

def _triangulate(indices, counts):
    """Fan-triangulate a face-vertex stream into a flat (3*T,) triangle index array.

    The manual scenes are already all-triangle, but Blender can export n-gons, and
    write_layout_usd assumes three indices per face -- so normalize defensively.
    """
    tris = []
    off = 0
    for c in counts:
        face = indices[off : off + c]
        for k in range(1, c - 1):
            tris.extend((int(face[0]), int(face[k]), int(face[k + 1])))
        off += c
    return np.array(tris, dtype=np.int32)


def _decompose(matrix):
    """TRS-decompose a world matrix into (pose[x,y,z,qx,qy,qz,qw], scale-vec).

    Blender authors each object as translate * rotateXYZ * scale, which Gf.Transform
    factors back out cleanly. write_layout_usd's pose carries only translation + rotation,
    so the caller bakes `scale` into the vertices.
    """
    xf = Gf.Transform(matrix)
    t = xf.GetTranslation()
    q = xf.GetRotation().GetQuat()
    im = q.GetImaginary()
    s = xf.GetScale()
    pose = np.array([t[0], t[1], t[2], im[0], im[1], im[2], q.GetReal()], dtype=float)
    return pose, np.array([s[0], s[1], s[2]], dtype=float)


def _body_from_mesh(prim, xform_cache):
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    if np.any(counts != 3):
        indices = _triangulate(indices, counts)

    pose, scale = _decompose(xform_cache.GetLocalToWorldTransform(prim))
    verts = (pts * scale).astype(np.float32)  # bake scale; pose holds only T + R

    name = prim.GetName()
    return UsdBody(
        name=name,
        vertices=verts,
        indices=indices,
        pose=pose,
        is_static=name.startswith(TABLE),  # Blender suffixes the table as dining_room_table_NNN
    )


def convert(scene_path, out_path):
    """Author the physics USD for one manual scene at an explicit output path.

    `scene_path` is the manual Scene*.usdc to read; `out_path` is the exact
    *_physics.usdc file to write. The caller owns the on-disk layout (e.g.
    data-set's <scene>/data/<scene>_physics.usdc); convert just writes there.
    Both args accept str or pathlib.Path. Returns the output Path."""
    scene_path, out_path = Path(scene_path), Path(out_path)
    stage = Usd.Stage.Open(str(scene_path))
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bodies = [
        _body_from_mesh(p, xform_cache)
        for p in stage.Traverse()
        if p.GetTypeName() == "Mesh"
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)  # write_layout_usd uses Usd.Stage.CreateNew, which won't overwrite
    write_layout_usd(str(out_path), bodies, gravity=abs(GRAVITY), friction=MU, restitution=RESTITUTION)

    n_static = sum(b.is_static for b in bodies)
    print(f"{scene_path.name}: {len(bodies)} bodies ({n_static} static) -> {out_path}")
    return out_path


if __name__ == "__main__":
    # Manual Blender exports are nested one per scene at manual/<Scene>/data/<Scene>.usdc
    # (not flat in manual/), so key each by its folder; stray flat re-exports are ignored.
    pass