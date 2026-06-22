"""Render the occluded target of each scene as a flat-color PNG.

The companion to :mod:`data_pipeline.render`: where ``run_render`` produces the
photoreal ``render.png`` + ``segmentation.png`` for a scene, this module answers
"what, and where, is the hidden object?" — it emits a ``target.png`` showing the
occluded target's *full* outline (occluders ignored), filled with the exact color
``render_scene`` generated the object with.

How the target is chosen (no per-scene annotation needed):

    The segmentation pass labels every tracked object a unique id; an object that
    is occluded from the camera leaves few or no pixels in ``segmentation.png``.
    So we project each object's geometry from the scene camera to get its full
    *unoccluded* silhouette, then compare against how many of those pixels actually
    survive in the segmentation. The least-visible non-table object is the target,
    and its ``visible_fraction`` (visible px / silhouette px) is recorded as the
    "little error" — 0.0 when fully hidden, small and positive when a sliver shows.

Why a flat fill rather than a Blender render: it needs no GPU/Blender (the camera
is reproduced exactly on the CPU — see :class:`_Pinhole`), so the silhouette lands
on the same pixel grid as ``render.png`` / ``segmentation.png`` (even when those
were supersampled), and the color is *exactly* the authored albedo rather than a
light-dependent shaded approximation.

    from data_pipeline import run_render_targets
    run_render_targets("data-set")                       # all rendered scenes
    run_render_targets("data-set", ["scene015"])         # a subset

The scene must already be rendered (``results/segmentation.png`` +
``segmentation_labels.json`` present); scenes missing those are skipped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Usd, UsdGeom

from scene_physics.configs.camera import CameraIntrinsics
from scene_physics.visualization.blender.material_specs import DEFAULT, MATERIAL_SPECS

from data_pipeline.utils.config import load_experiment_config
from data_pipeline.render.render import scene_dirs

__all__ = [
    "spec_for",
    "base_color_rgb",
    "TargetReport",
    "occluded_target",
    "render_target",
    "run_render_targets",
]

# The substrate the target sits on is never the occluded object; exclude it from the
# candidate set. Mirrors scene_physics.data_gen.scene_gen.TABLE (kept in sync by hand
# so this stays importable without pulling in warp/newton).
TABLE_PREFIX = "dining_room_table"


# --------------------------------------------------------------------------- color


def spec_for(name: str) -> dict:
    """The ``MATERIAL_SPECS`` entry for an object, matched by base name.

    Scene prims carry instance suffixes (``square_wood_block_013``,
    ``f10_apple_iphone_4_003_001``) and some base names themselves end in digits, so
    we match the *longest* spec key that prefixes ``name`` at an underscore boundary
    rather than stripping a trailing ``_NNN``. Falls back to ``DEFAULT`` (neutral
    gray) for objects with no spec — exactly the color ``render_scene`` gives them.
    """
    best: str | None = None
    for key in MATERIAL_SPECS:
        if (name == key or name.startswith(key + "_")) and (
            best is None or len(key) > len(best)
        ):
            best = key
    return MATERIAL_SPECS[best] if best is not None else DEFAULT


def base_color_rgb(name: str) -> tuple[int, int, int]:
    """8-bit ``(r, g, b)`` flat fill for ``name``: its ``MATERIAL_SPECS`` base_color
    (the albedo the scene was generated with), used directly as the display color
    (no view transform) so the fill is exactly the authored color."""
    base = spec_for(name).get("base_color", DEFAULT["base_color"])
    return tuple(int(round(c * 255)) for c in base[:3])


# ------------------------------------------------------------------------ geometry


@dataclass(frozen=True)
class _Pinhole:
    """The exact pinhole ``render_scene``'s Blender camera uses (see
    ``visualization/blender/_camera.py``): Z-up world looking down ``forward``,
    vertical FOV, square pixels, principal point at the image centre. Built at a
    given pixel resolution so a projected silhouette lands on the same grid as
    ``render.png`` / ``segmentation.png`` (which may have been supersampled)."""

    eye: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal_px: float
    width: int
    height: int

    @classmethod
    def from_intrinsics(
        cls, intr: CameraIntrinsics, width: int, height: int
    ) -> "_Pinhole":
        eye = np.asarray(intr.eye, dtype=float)
        forward = np.asarray(intr.target, dtype=float) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(intr.up, dtype=float))
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        focal_px = (height / 2) / np.tan(np.radians(float(intr.fov_degree)) / 2)
        return cls(eye, right, cam_up, forward, focal_px, int(width), int(height))

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World ``(N, 3)`` -> ``(pixels (N, 2), depth (N,))``; ``depth > 0`` is in
        front of the camera. ``+x`` right, ``+y`` down (image convention)."""
        d = np.asarray(points, dtype=float) - self.eye
        depth = d @ self.forward
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        px = self.width / 2 + self.focal_px * ((d @ self.right) / safe)
        py = self.height / 2 - self.focal_px * ((d @ self.up) / safe)
        return np.stack([px, py], axis=1), depth


def _object_meshes(usd_path: Path) -> tuple[Usd.Stage, dict, UsdGeom.XformCache]:
    """Open ``usd_path`` and return ``(stage, {name: Mesh prim}, xform_cache)``. The
    stage is returned so the caller keeps it alive while reading prim attributes."""
    stage = Usd.Stage.Open(str(usd_path))
    meshes = {p.GetName(): p for p in stage.Traverse() if p.GetTypeName() == "Mesh"}
    return stage, meshes, UsdGeom.XformCache(Usd.TimeCode.Default())


def _silhouette(prim, xform_cache: UsdGeom.XformCache, cam: _Pinhole) -> np.ndarray:
    """Boolean ``(H, W)`` mask of the prim's *full* projected silhouette (occluders
    ignored). Triangles with any vertex behind the camera are dropped — objects sit
    well in front, so this only guards the degenerate case."""
    mesh = UsdGeom.Mesh(prim)
    local = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    to_world = xform_cache.GetLocalToWorldTransform(prim)
    world = np.array([to_world.Transform(Gf.Vec3d(*p)) for p in local])
    pix, depth = cam.project(world)

    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    img = Image.new("L", (cam.width, cam.height), 0)
    draw = ImageDraw.Draw(img)
    off = 0
    for c in counts:
        face = indices[off : off + c]
        off += c
        if np.all(depth[face] > 1e-6):
            draw.polygon([(float(pix[i, 0]), float(pix[i, 1])) for i in face], fill=255)
    return np.asarray(img) > 0


# ------------------------------------------------------------------- target picking


@dataclass(frozen=True)
class TargetReport:
    """Per-object visibility from the scene camera. ``visible_fraction`` is
    ``visible_px / silhouette_px`` — 0.0 when no pixel survives in the segmentation
    (fully hidden), small and positive for a slightly-visible target."""

    name: str
    object_id: int
    visible_px: int
    silhouette_px: int
    visible_fraction: float
    fully_hidden: bool


def occluded_target(
    seg: np.ndarray,
    labels: dict,
    meshes: dict,
    xform_cache: UsdGeom.XformCache,
    cam: _Pinhole,
    *,
    table_prefix: str = TABLE_PREFIX,
) -> tuple[TargetReport | None, np.ndarray | None, list[TargetReport]]:
    """Pick the occluded target — the non-table object least visible in ``seg``
    relative to its full projected silhouette — and return
    ``(target, target_mask, all_reports)``."""
    reports: list[TargetReport] = []
    masks: dict[str, np.ndarray] = {}
    for name, oid in labels.items():
        prim = meshes.get(name)
        if name == "background" or name.startswith(table_prefix) or prim is None:
            continue
        mask = _silhouette(prim, xform_cache, cam)
        masks[name] = mask
        sil_px = int(mask.sum())
        vis_px = int((seg == oid).sum())
        frac = vis_px / sil_px if sil_px else 0.0
        reports.append(TargetReport(name, int(oid), vis_px, sil_px, frac, vis_px == 0))

    if not reports:
        return None, None, []
    # Least-visible object wins; ties (several objects fully hidden — a small prop
    # tucked behind something *and* the real target) break toward the LARGEST
    # silhouette, i.e. the substantive hidden object rather than an incidental trinket.
    winner = min(reports, key=lambda r: (r.visible_fraction, -r.silhouette_px))
    return winner, masks[winner.name], reports


# ----------------------------------------------------------------------- rendering


def render_target(
    scene_dir: str | Path,
    intr: CameraIntrinsics,
    *,
    out_dir: str | Path | None = None,
) -> dict:
    """Write ``target.png`` (+ ``target.json``) for one scene; return the metadata.

    Reads ``results/{segmentation.png, segmentation_labels.json}`` (the scene must
    already be rendered), picks the occluded target, and fills its full silhouette
    with its authored base color on a transparent background — at the segmentation's
    own resolution, so it overlays ``render.png`` exactly. ``out_dir`` redirects the
    PNG/JSON (default: the scene's ``results/``)."""
    scene_dir = Path(scene_dir)
    usd = next(scene_dir.glob("data/*_physics.usdc"))
    results = scene_dir / "results"
    seg_path = results / "segmentation.png"
    labels_path = results / "segmentation_labels.json"
    if not (seg_path.exists() and labels_path.exists()):
        raise FileNotFoundError(
            f"{scene_dir.name}: render it first — missing "
            f"results/segmentation.png or results/segmentation_labels.json"
        )

    out_dir = Path(out_dir) if out_dir is not None else results
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = np.asarray(Image.open(seg_path))
    height, width = seg.shape[:2]
    labels = json.loads(labels_path.read_text())
    cam = _Pinhole.from_intrinsics(intr, width, height)

    stage, meshes, xform_cache = _object_meshes(usd)
    winner, mask, reports = occluded_target(seg, labels, meshes, xform_cache, cam)
    del stage  # prim reads are done
    if winner is None:
        raise ValueError(f"{scene_dir.name}: no segmentable (non-table) objects")

    fill = base_color_rgb(winner.name)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask] = (*fill, 255)
    Image.fromarray(rgba, "RGBA").save(out_dir / "target.png")

    spec_color = spec_for(winner.name).get("base_color", DEFAULT["base_color"])
    meta = {
        "target": winner.name,
        "object_id": winner.object_id,
        "fully_hidden": winner.fully_hidden,
        "visible_px": winner.visible_px,
        "silhouette_px": winner.silhouette_px,
        "visible_fraction": round(winner.visible_fraction, 6),
        "base_color": [round(float(c), 4) for c in spec_color[:3]],
        "fill_rgb": list(fill),
        "resolution": [width, height],
        "objects": [asdict(r) for r in sorted(reports, key=lambda r: r.visible_fraction)],
    }
    (out_dir / "target.json").write_text(json.dumps(meta, indent=2))
    return meta


def run_render_targets(
    exp: str,
    scenes: list[str] | None = None,
    *,
    out_root: str | Path | None = None,
) -> list[tuple[str, dict]]:
    """Render the occluded-target PNG for every (already-rendered) scene in ``exp``.

    The viewpoint comes from ``config/<exp>.json`` (the same camera the scenes were
    rendered and occlusion-gated against). Scenes missing their segmentation outputs
    are skipped with a warning. Returns ``[(scene_name, metadata), ...]``.
    """
    intr = load_experiment_config(exp).camera
    dirs = scene_dirs(exp, scenes)
    if not dirs:
        raise FileNotFoundError(
            f"No scenes for exp={exp!r} "
            f"(pass scenes=['scene015', ...] or generate the dataset first)."
        )

    print(f"[run_render_targets] exp={exp} scenes={len(dirs)}")
    out: list[tuple[str, dict]] = []
    for d in dirs:
        scene_out = Path(out_root) / d.name if out_root is not None else None
        try:
            meta = render_target(d, intr, out_dir=scene_out)
        except FileNotFoundError as e:
            print(f"[run_render_targets] skip {e}")
            continue
        hidden = "fully hidden" if meta["fully_hidden"] else f"{meta['visible_fraction']:.3f} visible"
        print(
            f"[run_render_targets] {d.name} -> target={meta['target']} "
            f"({hidden}) fill={meta['fill_rgb']}"
        )
        out.append((d.name, meta))
    return out
