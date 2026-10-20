"""Plot overlapping train/validation force histograms from split CSVs.

Run from any directory using --data-dir to select a directory containing
train.csv and validation.csv. --separate also saves Fx, Fy, Fz and resultant
magnitude as four individual figures alongside the 2x2 overview.
Each bin shows its fraction of the combined train/validation sample count.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "Tactip_pic_data0716" / "Tactip_pic_data0716"
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_total_forces(index_path):
    paths = np.load(index_path)
    return np.asarray([np.load(path).reshape(-1, 3).sum(axis=0) for path in paths])


def load_csv_forces(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = ("force_fx", "force_fy", "force_fz")
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        forces = np.asarray(
            [[float(row[column]) for column in columns] for row in reader],
            dtype=float,
        )
    if not len(forces) or not np.isfinite(forces).all():
        raise ValueError(f"{csv_path} has no samples or contains non-finite forces")
    return forces


def load_node_forces(index_path):
    return np.asarray([np.load(path).reshape(16, 3) for path in np.load(index_path)])


def node_coverage(labels, contact_threshold_n=0.05):
    labels = np.asarray(labels)
    active = -labels[:, :, 2] >= contact_threshold_n
    magnitude = np.linalg.norm(labels, axis=2)
    active_count = active.sum(axis=0)
    active_rate = active.mean(axis=0)
    mean_force = np.divide(
        (magnitude * active).sum(axis=0),
        active_count,
        out=np.zeros(labels.shape[1], dtype=float),
        where=active_count > 0,
    )
    return active_rate, mean_force


def node_fx_fy_histograms(
    labels,
    bins=30,
    contact_threshold_n=0.05,
    tangential_threshold_n=0.05,
    force_limit=None,
):
    labels = np.asarray(labels)
    if labels.ndim != 3 or labels.shape[1:] != (16, 3):
        raise ValueError("labels must have shape [samples, 16, 3]")
    tangential = np.linalg.norm(labels[:, :, :2], axis=2)
    active = (-labels[:, :, 2] >= contact_threshold_n) & (
        tangential >= tangential_threshold_n
    )
    if force_limit is None:
        values = labels[:, :, :2][active]
        force_limit = max(0.1, float(np.abs(values).max())) if values.size else 0.1
    if force_limit <= 0:
        raise ValueError("force_limit must be positive")

    edges = np.linspace(-force_limit, force_limit, bins + 1)
    histograms = np.zeros((16, bins, bins), dtype=float)
    counts = active.sum(axis=0)
    for node in range(16):
        values = labels[active[:, node], node, :2]
        if len(values):
            histogram, _, _ = np.histogram2d(
                values[:, 0], values[:, 1], bins=(edges, edges)
            )
            histograms[node] = histogram.T / len(values)
    return histograms, edges, counts


def train_index_path(data_dir, train_variant="balanced"):
    suffix = f"_{train_variant}" if train_variant else ""
    return Path(data_dir) / f"train{suffix}_node_force.npy"


def validation_index_path(data_dir, validation_variant="balanced"):
    suffix = f"_{validation_variant}" if validation_variant else ""
    return Path(data_dir) / f"validation{suffix}_node_force.npy"


def spatial_index_paths(
    data_dir, train_variant="balanced", validation_variant="balanced"
):
    return (
        train_index_path(data_dir, train_variant),
        validation_index_path(data_dir, validation_variant),
    )


def distribution_values(forces):
    return (
        [forces[:, 0], forces[:, 1], forces[:, 2], np.linalg.norm(forces, axis=1)],
        ("Fx", "Fy", "Fz", "Resultant magnitude"),
    )


def distribution_weights(train_count, validation_count):
    weight = 1.0 / (train_count + validation_count)
    return np.full(train_count, weight), np.full(validation_count, weight)


def _draw_distribution(axis, train_value, validation_value, name, bins, weights, labels):
    edges = np.histogram_bin_edges(np.concatenate((train_value, validation_value)), bins=bins)
    for values, sample_weights, label, color in zip(
        (train_value, validation_value), weights, labels, ("#1f77b4", "#ff7f0e")
    ):
        axis.hist(
            values, bins=edges, weights=sample_weights,
            histtype="stepfilled", alpha=0.55, color=color,
            edgecolor=color, linewidth=1.0, label=label,
        )
    axis.set(title=name, xlabel="Force (N)", ylabel="占总数据的比例")
    axis.set_ylim(bottom=0.0)
    axis.set_axisbelow(True)
    axis.grid(alpha=0.2)
    axis.legend()


def plot_distributions(
    data_dir,
    output=None,
    bins=50,
    train_variant="balanced",
    validation_variant="balanced",
    separate=False,
):
    if bins <= 0:
        raise ValueError("bins must be a positive integer")
    data_dir = Path(data_dir)
    csv_splits = (data_dir / "train.csv", data_dir / "validation.csv")
    use_csv = any(path.is_file() for path in csv_splits)
    if use_csv:
        train, validation = (load_csv_forces(path) for path in csv_splits)
        train_name, validation_name = "Train", "Validation"
    else:
        train = load_total_forces(train_index_path(data_dir, train_variant))
        validation = load_total_forces(
            validation_index_path(data_dir, validation_variant)
        )
        train_name = "Train" + (f" ({train_variant})" if train_variant else "")
        validation_name = "Validation" + (
            f" ({validation_variant})" if validation_variant else ""
        )
    values, names = distribution_values(train)
    val_values, _ = distribution_values(validation)
    train_weights, validation_weights = distribution_weights(
        len(train), len(validation)
    )

    weights = (train_weights, validation_weights)
    labels = (f"{train_name} (n={len(train)})", f"{validation_name} (n={len(validation)})")
    title = f"Train vs Validation Force Distributions (total n={len(train) + len(validation)})"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, train_value, val_value, name in zip(axes.flat, values, val_values, names):
        _draw_distribution(axis, train_value, val_value, name, bins, weights, labels)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    train_suffix = f"_{train_variant}" if train_variant else ""
    validation_suffix = f"_{validation_variant}" if validation_variant else ""
    output = Path(output) if output else data_dir / (
        "train_validation_force_distribution.png"
        if use_csv
        else f"train{train_suffix}_validation{validation_suffix}_distribution.png"
    )
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if separate:
        for train_value, val_value, name, suffix in zip(
            values, val_values, names, ("fx", "fy", "fz", "resultant")
        ):
            panel, axis = plt.subplots(figsize=(7, 5))
            _draw_distribution(axis, train_value, val_value, name, bins, weights, labels)
            panel.suptitle(f"Train vs Validation (total n={len(train) + len(validation)})")
            panel.tight_layout()
            panel.savefig(
                output.with_name(f"{output.stem}_{suffix}{output.suffix}"),
                dpi=180, bbox_inches="tight",
            )
            plt.close(panel)
    return output


def plot_spatial_coverage(
    data_dir,
    output=None,
    contact_threshold_n=0.05,
    train_variant="balanced",
    validation_variant="balanced",
):
    data_dir = Path(data_dir)
    train_path, validation_path = spatial_index_paths(
        data_dir, train_variant, validation_variant
    )
    train = node_coverage(
        load_node_forces(train_path), contact_threshold_n
    )
    validation = node_coverage(
        load_node_forces(validation_path), contact_threshold_n
    )
    train_name = "Train" + (f" ({train_variant})" if train_variant else "")
    validation_name = "Validation" + (
        f" ({validation_variant})" if validation_variant else ""
    )
    node_ids = np.arange(1, 17).reshape(4, 4)[:, ::-1]
    coordinates = (-12.9, -4.3, 4.3, 12.9)
    panels = (
        (train[0], f"{train_name} active rate", "%"),
        (validation[0], f"{validation_name} active rate", "%"),
        (train[1], f"{train_name}\nmean force when active", "N"),
        (validation[1], f"{validation_name}\nmean force when active", "N"),
    )
    limits = (max(train[0].max(), validation[0].max()),) * 2 + (
        max(train[1].max(), validation[1].max()),
    ) * 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    images = []
    for axis, (values, title, unit), vmax in zip(axes.flat, panels, limits):
        grid = values.reshape(4, 4)[:, ::-1]
        image = axis.imshow(grid, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
        images.append(image)
        for row in range(4):
            for column in range(4):
                value = grid[row, column]
                label = f"{value:.1%}" if unit == "%" else f"{value:.2f} N"
                axis.text(
                    column,
                    row,
                    f"N{node_ids[row, column]}\n{label}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 * vmax else "black",
                    fontsize=9,
                )
        axis.set_title(title)
        axis.set_xticks(range(4), coordinates)
        axis.set_yticks(range(4), coordinates)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")

    fig.colorbar(images[0], ax=axes[0], label="Active sample fraction", shrink=0.85)
    fig.colorbar(images[2], ax=axes[1], label="Mean force magnitude (N)", shrink=0.85)
    fig.suptitle(f"16-node Spatial Force Coverage (-Fz >= {contact_threshold_n:g} N)")
    train_suffix = f"_{train_variant}" if train_variant else ""
    validation_suffix = f"_{validation_variant}" if validation_variant else ""
    output = Path(output) if output else data_dir / (
        f"train{train_suffix}_validation{validation_suffix}_spatial_coverage.png"
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_node_fx_fy_distribution(
    data_dir,
    split="train",
    variant="balanced",
    output=None,
    bins=30,
    contact_threshold_n=0.05,
    tangential_threshold_n=0.05,
):
    data_dir = Path(data_dir)
    if split not in ("train", "validation"):
        raise ValueError("split must be 'train' or 'validation'")
    index_path = (
        train_index_path(data_dir, variant)
        if split == "train"
        else validation_index_path(data_dir, variant)
    )
    histograms, edges, counts = node_fx_fy_histograms(
        load_node_forces(index_path),
        bins=bins,
        contact_threshold_n=contact_threshold_n,
        tangential_threshold_n=tangential_threshold_n,
    )

    node_ids = np.arange(1, 17).reshape(4, 4)[:, ::-1][::-1]
    vmax = max(float(histograms.max()), np.finfo(float).eps)
    fig, axes = plt.subplots(4, 4, figsize=(11, 10), sharex=True, sharey=True)
    image = None
    for axis, node_id in zip(axes.flat, node_ids.flat):
        image = axis.imshow(
            histograms[node_id - 1],
            origin="lower",
            extent=(edges[0], edges[-1], edges[0], edges[-1]),
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
            aspect="equal",
        )
        axis.axhline(0.0, color="white", linewidth=0.5, alpha=0.5)
        axis.axvline(0.0, color="white", linewidth=0.5, alpha=0.5)
        axis.set_title(f"N{node_id} (n={counts[node_id - 1]})", fontsize=10)

    fig.supxlabel("Fx (N)")
    fig.supylabel("Fy (N)")
    fig.suptitle(
        f"{split.title()} ({variant}) per-node Fx-Fy distributions\n"
        f"-Fz >= {contact_threshold_n:g} N, |Ft| >= {tangential_threshold_n:g} N"
    )
    fig.colorbar(image, ax=axes, label="Fraction of active samples per node", shrink=0.8)
    output = Path(output) if output else data_dir / (
        f"{split}_{variant}_node_fx_fy_distribution.png"
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main(cli_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing train.csv and validation.csv")
    parser.add_argument("--output")
    parser.add_argument("--bins", type=int,
                        help="Number of bins (default: 50 for global forces, 30 for node Fx-Fy)")
    parser.add_argument("--separate", action="store_true",
                        help="Also export four individual global-force distribution figures")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--spatial", action="store_true")
    plot_group.add_argument("--node-fx-fy", action="store_true")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument(
        "--train-variant",
        default="balanced",
        help="Training index variant (default: balanced)",
    )
    parser.add_argument(
        "--validation-variant",
        default="balanced",
        help="Validation index variant (default: balanced)",
    )
    args = parser.parse_args(cli_args)
    if args.node_fx_fy:
        variant = (
            args.train_variant if args.split == "train" else args.validation_variant
        )
        print(
            plot_node_fx_fy_distribution(
                args.data_dir,
                split=args.split,
                variant=variant,
                output=args.output,
                bins=30 if args.bins is None else args.bins,
            )
        )
    elif args.spatial:
        print(
            plot_spatial_coverage(
                args.data_dir,
                args.output,
                train_variant=args.train_variant,
                validation_variant=args.validation_variant,
            )
        )
    else:
        print(
            plot_distributions(
                args.data_dir,
                args.output,
                bins=50 if args.bins is None else args.bins,
                train_variant=args.train_variant,
                validation_variant=args.validation_variant,
                separate=args.separate,
            )
        )


if __name__ == "__main__":
    main()
