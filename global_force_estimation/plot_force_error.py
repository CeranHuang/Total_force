"""Plot signed test-force errors grouped by ground-truth force, in newtons.

For an existing model, run from the project root::

    python global_force_estimation/plot_force_error.py saved_models/global_include_zero

Inputs are paired, denormalized [N, 3] arrays in Fx, Fy, Fz order. No force
rotation, sign reversal, clipping or display deadband is applied.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_force_error_plot(predictions, targets, output_dir):
    """Save a four-panel PNG and per-range statistics CSV; return the PNG path.

    The last panel uses norm(prediction) - norm(target), which measures
    magnitude error rather than the direction-sensitive norm(prediction-target).
    Fixed interior boundaries permit comparison across runs; open outer bins
    retain outliers and positive Fz samples without changing the sensor signs.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if (predictions.shape != targets.shape or predictions.ndim != 2
            or predictions.shape[1] != 3 or len(predictions) == 0):
        raise ValueError("Predictions and targets must have equal nonempty [N, 3] shapes.")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Predictions and targets must contain only finite forces.")

    truth = np.column_stack((targets, np.linalg.norm(targets, axis=1)))
    errors = np.column_stack((
        predictions - targets,
        np.linalg.norm(predictions, axis=1) - truth[:, 3],
    ))
    panels = (
        ("Fx", "Fx", (-1.5, 0., 1.5), "#258d8f"),
        ("Fy", "Fy", (-1.5, 0., 1.5), "#7544a0"),
        ("Fz", "Fz (compression < 0)", (-7.5, -5., -2.5), "#dc3366"),
        ("Magnitude", "Resultant magnitude", (2.5, 5., 7.5), "#ef6c43"),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "force_error_distribution.png"
    rows = []
    rng = np.random.default_rng(42)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    try:
        for column, (axis, (component, title, cuts, color)) in enumerate(zip(axes.flat, panels)):
            groups = np.digitize(truth[:, column], cuts)
            edges = (-np.inf, *cuts, np.inf)
            labels = []
            for group, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                values = errors[groups == group, column]
                count = len(values)
                interval = (f"< {high:g}" if group == 0 else
                            f">= {low:g}" if group == 3 else f"[{low:g}, {high:g})")
                labels.append(f"{interval}\nn={count}")
                rows.append({
                    "component": component, "lower_n": low, "upper_n": high,
                    "count": count,
                    "bias_n": float(values.mean()) if count else "",
                    "median_n": float(np.median(values)) if count else "",
                    "mae_n": float(np.abs(values).mean()) if count else "",
                    "rmse_n": float(np.sqrt(np.square(values).mean())) if count else "",
                })
                if not count:
                    continue
                # Sparse/constant samples do not support a useful density estimate.
                if count >= 20 and np.ptp(values) > 1e-10:
                    violin = axis.violinplot([values], positions=[group], widths=0.72,
                                             showextrema=False)
                    body = violin["bodies"][0]
                    vertices = body.get_paths()[0].vertices
                    vertices[:, 0] = np.maximum(vertices[:, 0], group)
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.55)
                axis.scatter(group + rng.uniform(-0.27, -0.04, count), values,
                             s=10, color=color, alpha=0.5, edgecolors="none", zorder=3)
                median = np.median(values)
                axis.plot([group - 0.06, group + 0.16], [median, median],
                          color="#222222", linewidth=2.2, zorder=4)

            axis.axhline(0., color="#444444", linestyle="--", linewidth=0.9)
            axis.set_title(f"{'abcd'[column]}   {title}", loc="left", fontweight="semibold")
            axis.set_xticks(range(4), labels)
            axis.set_xlim(-0.5, 3.5)
            axis.set_xlabel("Ground-truth force range (N)")
            axis.set_ylabel("Signed force error (N)")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            axis.margins(y=0.15)
            axis.spines[["top", "right"]].set_visible(False)

        figure.suptitle(f"Test-set force prediction errors  (n={len(targets)})", fontsize=14)
        figure.text(0.5, 0.055,
                    "Dots: samples   |   Black bars: medians   |   Half-violins: density (n >= 20)",
                    ha="center", fontsize=9)
        figure.text(0.5, 0.025,
                    "Error = prediction - truth; resultant error = ||prediction|| - ||truth|| (magnitude only).",
                    ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.09, 1, 0.95))
        figure.savefig(plot_path, dpi=300)
    finally:
        plt.close(figure)

    with (output_dir / "force_error_by_range.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return plot_path


def main(cli_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path,
                        help="Model directory containing test_predictions.npy and test_targets.npy")
    args = parser.parse_args(cli_args)
    path = save_force_error_plot(
        np.load(args.model_dir / "test_predictions.npy", allow_pickle=False),
        np.load(args.model_dir / "test_targets.npy", allow_pickle=False),
        args.model_dir / "plots",
    )
    print(f"Force-error plot: {path}")


if __name__ == "__main__":
    main()
