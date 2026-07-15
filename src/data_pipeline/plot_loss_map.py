"""Plot a scene's loss map: per-channel softmax panels plus combined figures.

Companion viewer for ``data_pipeline.loss_map``: reads the
``results/loss_map.npy`` that CLI saves for a ``data/<exp>/<scene>``. The map
is ``(ny, nx, 4)`` raw per-location scores from
``ParallelPhysicsLikelihood.non_rank_physics`` — columns
``[still, -penetration, -displacement, -rotation]``, every column oriented so
higher = more likely — and normalization is deferred to plotting time via a
softmax over the whole grid (nothing is batch-relative).

Figures (all land in the scene's ``results/``):

- ``loss_map_channels.png`` — softmax of each channel separately, plus a
  fifth panel: ``softmax(sum_k lambda_k * channel_k)`` with the lambdas given
  on the CLI.
- ``loss_map_2d.png`` / ``loss_map_3d.png`` — the combined softmax map,
  raw next to a copy whose *logits* were Gaussian-smoothed first (the
  per-sample physics scores are noisy; smoothing before the softmax exposes
  the structure).

Each panel marks the hidden object's true position and its own argmax::

    python -m data_pipeline.plot_loss_map --exp data-set --scene scene001 \\
        --lambdas 1 1 1 1

Pure matplotlib over saved arrays — this module imports nothing from
scene_physics (though ``data_pipeline/__init__.py`` currently pulls in warp
via ``bulk_scene`` for any ``-m data_pipeline.*`` launch).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from data_pipeline.utils.paths import data_dir

__all__ = ["load_map", "softmax", "plot_channels", "plot_2d", "plot_3d",
           "plot_scene"]

# Column order of non_rank_physics' output (signs already baked in upstream).
CHANNELS = ("still render", "penetration", "displacement", "rotation")

TRUTH_STYLE = dict(color="#c2410c", marker="o")
PEAK_STYLE = dict(color="#1d4ed8", marker="^")


def load_map(exp: str, scene: str) -> tuple:
    """Load one scene's loss map and the metadata needed to plot it.

    Returns ``(name, xs, ys, C, truth_xy, results_dir)`` — the hidden object's
    name, the grid's world coordinates (``C`` rows are y, columns are x,
    matching ``gen_surface``'s ``np.meshgrid``), the ``(ny, nx, 4)`` raw score
    channels, the object's true (x, y), and the scene's ``results/`` dir.

    Raises ``FileNotFoundError`` for a missing input, naming it; the fix for a
    missing ``loss_map.npy`` is running ``data_pipeline.loss_map`` first.
    """
    scene_dir = data_dir(exp) / scene
    npy = scene_dir / "results" / "loss_map.npy"
    priors_json = scene_dir / "data" / f"{scene}_priors.json"
    truth_json = scene_dir / "data" / f"{scene}_truth.json"

    if not npy.exists():
        raise FileNotFoundError(
            f"{exp}/{scene}: no {npy} — generate it first "
            f"(python -m data_pipeline.loss_map --exp {exp} --scene {scene})."
        )
    missing = [p.name for p in (priors_json, truth_json) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{exp}/{scene}: missing {', '.join(missing)} in {scene_dir / 'data'}."
        )

    C = np.load(npy)
    if C.ndim != 3 or C.shape[-1] != len(CHANNELS):
        raise ValueError(
            f"{exp}/{scene}: loss_map.npy has shape {C.shape}, expected "
            f"(ny, nx, {len(CHANNELS)}) — a legacy rank-based map; regenerate "
            f"it (python -m data_pipeline.loss_map --exp {exp} --scene {scene})."
        )

    priors = json.loads(priors_json.read_text())
    if len(priors) != 1:
        raise ValueError(
            f"{exp}/{scene}: expected exactly one hidden object in priors, "
            f"found {sorted(priors)}."
        )
    (name, prior), = priors.items()

    # Rebuild gen_surface's grid from the array shape instead of duplicating
    # its FIDELITY constant (rows sweep y, columns sweep x).
    ny, nx = C.shape[:2]
    xs = prior["x_min"] + (prior["x_max"] - prior["x_min"]) / nx * np.arange(nx)
    ys = prior["y_min"] + (prior["y_max"] - prior["y_min"]) / ny * np.arange(ny)

    truth = json.loads(truth_json.read_text())[name]
    return name, xs, ys, C, (truth[0], truth[1]), scene_dir / "results"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the entire grid (shape-preserving, overflow-safe)."""
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _peak(xs, ys, data) -> tuple:
    i, j = np.unravel_index(np.argmax(data), data.shape)
    return xs[j], ys[i]


def _heatmap(ax, xs, ys, data, truth_xy, label):
    """One softmax heatmap panel with truth/argmax markers; returns the image."""
    extent = (xs[0], 2 * xs[-1] - xs[-2], ys[0], 2 * ys[-1] - ys[-2])
    im = ax.imshow(
        data, origin="lower", extent=extent, cmap="Blues",
        interpolation="nearest", aspect="equal",
    )
    x_true, y_true = truth_xy
    ax.plot(x_true, y_true, ls="", ms=8, mec="white", mew=1.5,
            label=f"true ({x_true:.2f}, {y_true:.2f})", **TRUTH_STYLE)
    if np.ptp(data) > 0:  # a flat channel has no meaningful argmax
        x_pk, y_pk = _peak(xs, ys, data)
        ax.plot(x_pk, y_pk, ls="", ms=8, mec="white", mew=1.5,
                label=f"peak ({x_pk:.2f}, {y_pk:.2f})", **PEAK_STYLE)
    else:
        label += " (flat)"
    ax.set_title(label, fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    return im


def _colorbar(fig, im, ax, **kw):
    """Colorbar in scientific notation (softmax probabilities are tiny)."""
    cb = fig.colorbar(im, ax=ax, label="probability", **kw)
    cb.formatter.set_powerlimits((-2, 3))
    cb.update_ticks()
    return cb


def plot_channels(title, xs, ys, C, truth_xy, out_path, *, lambdas):
    """Softmax of each raw channel plus the lambda-combined map; saves PNG."""
    lambdas = np.asarray(lambdas, dtype=float)
    lam_txt = ", ".join(f"{l:g}" for l in lambdas)
    panels = [(name, softmax(C[..., k])) for k, name in enumerate(CHANNELS)]
    panels.append((f"combined — softmax(Σ λ·channel), λ = ({lam_txt})",
                   softmax((C * lambdas).sum(axis=-1))))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    for ax, (label, prob) in zip(axes.flat, panels):
        im = _heatmap(ax, xs, ys, prob, truth_xy, label)
        _colorbar(fig, im, ax, shrink=0.8)
    axes.flat[-1].set_axis_off()

    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    for ax in axes[1, :]:
        ax.set_xlabel("x (m)")

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _combined_panels(C, lambdas, sigma):
    """(label, probability) pairs: plain softmax and softmax of smoothed logits."""
    logits = (C * np.asarray(lambdas, dtype=float)).sum(axis=-1)
    return [
        ("softmax", softmax(logits)),
        (f"softmax of smoothed logits (σ = {sigma:g} cm)",
         softmax(gaussian_filter(logits, sigma=sigma))),
    ]


def plot_2d(title, xs, ys, C, truth_xy, out_path, *, lambdas,
            sigma: float = 2.0):
    """Two-panel heatmap of the combined map, intensity = probability."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                             layout="constrained")
    for ax, (label, prob) in zip(axes, _combined_panels(C, lambdas, sigma)):
        im = _heatmap(ax, xs, ys, prob, truth_xy, label)
        _colorbar(fig, im, ax, shrink=0.75)
        ax.set_xlabel("x (m)")

    axes[0].set_ylabel("y (m)")
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_3d(title, xs, ys, C, truth_xy, out_path, *, lambdas,
            sigma: float = 2.0):
    """Two-panel 3-D surface of the combined map with floor contours."""
    x_true, y_true = truth_xy
    X, Y = np.meshgrid(xs, ys)
    i_t = np.abs(ys - y_true).argmin()
    j_t = np.abs(xs - x_true).argmin()

    fig = plt.figure(figsize=(16, 7.5))
    for k, (label, prob) in enumerate(_combined_panels(C, lambdas, sigma)):
        floor = -0.15 * prob.max()
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(
            X, Y, prob, cmap="Blues", rstride=1, cstride=1,
            linewidth=0, antialiased=True,
        )
        ax.contourf(X, Y, prob, levels=12, cmap="Blues", alpha=0.55,
                    offset=floor)

        # True position as a stem from the floor up to the surface.
        z_t = prob[i_t, j_t]
        ax.plot([x_true, x_true], [y_true, y_true], [floor, z_t],
                color=TRUTH_STYLE["color"], lw=1.5, ls="--", zorder=10)
        ax.scatter([x_true], [y_true], [z_t], s=45, zorder=11,
                   label=f"true position ({x_true:.2f}, {y_true:.2f})",
                   **TRUTH_STYLE)
        x_pk, y_pk = _peak(xs, ys, prob)
        ax.scatter([x_pk], [y_pk], [prob.max()], s=45, zorder=11,
                   label=f"peak ({x_pk:.2f}, {y_pk:.2f})", **PEAK_STYLE)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("probability")
        ax.set_zlim(floor, prob.max())
        ax.set_title(label)
        ax.view_init(elev=32, azim=-125)
        ax.legend(loc="upper left", frameon=False)

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scene(exp: str, scene: str, *, kind: str = "all",
               lambdas=(1.0, 1.0, 1.0, 1.0), sigma: float = 2.0) -> list:
    """Render the requested figure(s) for one ``data/<exp>/<scene>``; return
    the saved paths."""
    name, xs, ys, C, truth_xy, results = load_map(exp, scene)
    title = f"{scene} — physics likelihood over hidden-object prior ({name})"

    saved = []
    if kind in ("channels", "all"):
        saved.append(plot_channels(title, xs, ys, C, truth_xy,
                                   results / "loss_map_channels.png",
                                   lambdas=lambdas))
    if kind in ("2d", "all"):
        saved.append(plot_2d(title, xs, ys, C, truth_xy,
                             results / "loss_map_2d.png",
                             lambdas=lambdas, sigma=sigma))
    if kind in ("3d", "all"):
        saved.append(plot_3d(title, xs, ys, C, truth_xy,
                             results / "loss_map_3d.png",
                             lambdas=lambdas, sigma=sigma))
    return saved


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot the loss map of one data/<exp>/<scene> "
                    "(run data_pipeline.loss_map first).",
    )
    ap.add_argument(
        "--exp", help="experiment id, e.g. data-set", required=True
        )

    ap.add_argument(
        "--scene", help="scene name, e.g. scene001", required=True
        )

    ap.add_argument(
        "--kind", choices=("channels", "2d", "3d", "all"), default="all",
        help="which figure(s) to render (default: all)",
        )

    ap.add_argument(
        "--lambdas", type=float, nargs=len(CHANNELS), default=[1.0] * len(CHANNELS),
        metavar=("L_STILL", "L_PEN", "L_DISP", "L_ROT"),
        help="weights combining the channels before the final softmax "
             "(default: 1 1 1 1)",
        )

    ap.add_argument(
        "--sigma", type=float, default=2.0,
        help="Gaussian smoothing of the combined logits for the right-hand "
             "panel, in grid cells (1 cell = FIDELITY = 1 cm; default: 2)",
        )
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    for path in plot_scene(args.exp, args.scene, kind=args.kind,
                           lambdas=args.lambdas, sigma=args.sigma):
        print(f"Saved {path}")
