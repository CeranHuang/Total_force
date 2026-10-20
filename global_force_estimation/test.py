"""Run a checkpoint on the test split and fit prediction-versus-GT lines."""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from force_estimation.metrics import COMPONENT_NAMES
from force_estimation.train import ForceEstimation


def fit_line(ground_truth, prediction):
    """Fit ``prediction = slope * ground_truth + intercept``."""
    ground_truth = np.asarray(ground_truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(ground_truth) != len(prediction) or len(ground_truth) < 2:
        raise ValueError("ground_truth and prediction need equal length >= 2.")
    centered_truth = ground_truth - ground_truth.mean()
    denominator = float(np.dot(centered_truth, centered_truth))
    if denominator == 0.0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    slope = float(np.dot(centered_truth, prediction - prediction.mean()) / denominator)
    intercept = float(prediction.mean() - slope * ground_truth.mean())
    fitted = slope * ground_truth + intercept
    residual_sum = float(np.square(prediction - fitted).sum())
    total_sum = float(np.square(prediction - prediction.mean()).sum())
    r2 = 1.0 - residual_sum / total_sum if total_sum else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2}


def save_force_regression(predictions, truths, output_dir):
    predictions = np.asarray(predictions, dtype=float)
    truths = np.asarray(truths, dtype=float)
    if predictions.shape != truths.shape:
        raise ValueError("predictions and truths must have equal shapes.")
    if predictions.ndim == 2 and predictions.shape[1] == 3:
        predicted_totals = predictions
        true_totals = truths
    elif predictions.ndim == 4 and predictions.shape[-1] == 3:
        predicted_totals = predictions.sum(axis=(1, 2))
        true_totals = truths.sum(axis=(1, 2))
    else:
        raise ValueError(
            "predictions and truths must have equal [N,3] or [N,H,W,3] shapes."
        )
    regressions = {
        name: fit_line(true_totals[:, index], predicted_totals[:, index])
        for index, name in enumerate(COMPONENT_NAMES)
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "force_regression.json"
    json_path.write_text(json.dumps(regressions, indent=2, allow_nan=True), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes, COMPONENT_NAMES)):
        ground_truth = true_totals[:, index]
        prediction = predicted_totals[:, index]
        result = regressions[name]
        low = min(float(ground_truth.min()), float(prediction.min()))
        high = max(float(ground_truth.max()), float(prediction.max()))
        if low == high:
            low, high = low - 0.5, high + 0.5
        line_x = np.array([low, high])
        axis.scatter(ground_truth, prediction, s=14, alpha=0.55)
        axis.plot(line_x, line_x, "k--", linewidth=1, label="Ideal: y=x")
        axis.plot(
            line_x,
            result["slope"] * line_x + result["intercept"],
            color="tab:red",
            linewidth=1.5,
            label="Linear fit",
        )
        axis.set(
            title=(f"{name}: slope={result['slope']:.3f}, "
                   f"intercept={result['intercept']:.3f}, R²={result['r2']:.3f}"),
            xlabel="Ground truth total force (N)",
            ylabel="Predicted total force (N)",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    plot_path = output_dir / "force_regression.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return regressions, plot_path, json_path


def save_global_force_metrics(predictions, truths, output_dir):
    """Save per-axis error metrics and a compact global-force dashboard."""
    predictions = np.asarray(predictions, dtype=float)
    truths = np.asarray(truths, dtype=float)
    if predictions.shape != truths.shape or predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("Global-force predictions and truths must have equal [N,3] shapes.")
    if predictions.shape[0] == 0:
        raise ValueError("At least one test sample is required to calculate metrics.")

    errors = predictions - truths
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    bias = np.mean(errors, axis=0)
    truth_variance = np.sum(np.square(truths - truths.mean(axis=0)), axis=0)
    residual_sum = np.sum(np.square(errors), axis=0)
    r2 = np.full(3, np.nan, dtype=float)
    valid_axes = truth_variance > 0.0
    r2[valid_axes] = 1.0 - residual_sum[valid_axes] / truth_variance[valid_axes]

    metrics = {
        "sample_count": int(predictions.shape[0]),
        "error_definition": "prediction - ground_truth",
        "mae_n": {name: float(mae[i]) for i, name in enumerate(COMPONENT_NAMES)},
        "rmse_n": {name: float(rmse[i]) for i, name in enumerate(COMPONENT_NAMES)},
        "bias_n": {name: float(bias[i]) for i, name in enumerate(COMPONENT_NAMES)},
        "r2": {name: float(r2[i]) for i, name in enumerate(COMPONENT_NAMES)},
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "force_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    positions = np.arange(len(COMPONENT_NAMES))
    bar_width = 0.36
    axes[0].bar(positions - bar_width / 2, mae, bar_width, label="MAE")
    axes[0].bar(positions + bar_width / 2, rmse, bar_width, label="RMSE")
    axes[0].set(
        title="Per-axis force error",
        ylabel="Force error (N)",
        xticks=positions,
        xticklabels=COMPONENT_NAMES,
    )
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(positions, r2, color="tab:purple")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        title="Predictive R² by axis",
        ylabel="R²",
        xticks=positions,
        xticklabels=[
            f"{name}\nbias={bias[i]:.2f} N"
            for i, name in enumerate(COMPONENT_NAMES)
        ],
    )
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].boxplot(
        [errors[:, i] for i in range(3)],
        tick_labels=COMPONENT_NAMES,
        showfliers=True,
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set(title="Signed error distribution", ylabel="Prediction − truth (N)")
    axes[2].grid(axis="y", alpha=0.3)

    plot_path = output_dir / "force_metrics.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metrics, plot_path, json_path


def config_for_checkpoint(config, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    model_match = re.fullmatch(r"(Resnet|Densenet)-(\d+)(_.+)", checkpoint.parent.name)
    epoch_match = re.fullmatch(r"epoch_(\d+)\.pt", checkpoint.name)
    if not checkpoint.is_file() or not model_match or not epoch_match:
        raise ValueError(
            "Expected saved_models/Resnet-18_<run>/epoch_<n>.pt or "
            "saved_models/global_<timestamp>/best.pt."
        )
    model_name, layer, run_suffix = model_match.groups()
    epoch = int(epoch_match.group(1))
    config = dict(config)
    config.update({
        "model_list": [[model_name, int(layer)]],
        "model_choice": 0,
        "weights": [[run_suffix, epoch]],
        "save_dir": str(checkpoint.parent.parent),
        "train_mode": False,
    })
    return config, epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="saved_models/Resnet-18_09-21_15-25-22/epoch_2.pt",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("force_config.yaml")),
    )
    parser.add_argument("--cuda-index", type=int)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    checkpoint = Path(args.checkpoint).resolve()
    is_global_checkpoint = (
        checkpoint.name == "best.pt"
        and re.fullmatch(r"global_\d{8}_\d{6}", checkpoint.parent.name)
        and checkpoint.is_file()
    )

    if is_global_checkpoint:
        result_dir = checkpoint.parent
        prediction_path = result_dir / "test_predictions.npy"
        truth_path = result_dir / "test_targets.npy"
        if not prediction_path.is_file() or not truth_path.is_file():
            raise FileNotFoundError(
                f"Global checkpoint test outputs are missing under {result_dir}. "
                "Run train_global.py to completion first."
            )
        predictions = np.load(prediction_path)
        truths = np.load(truth_path)
        output_dir = result_dir / "test_analysis"
        metrics, metrics_plot_path, metrics_json_path = save_global_force_metrics(
            predictions,
            truths,
            output_dir,
        )
        print(json.dumps(metrics, indent=2, allow_nan=True))
        print(f"Force metrics plot: {metrics_plot_path}")
        print(f"Force metrics values: {metrics_json_path}")
    else:
        config, epoch = config_for_checkpoint(config, checkpoint)
        cli_args = [] if args.cuda_index is None else ["--cuda_index", str(args.cuda_index)]
        evaluator = ForceEstimation(config, cli_args=cli_args)
        result_dir = evaluator.save_dir / f"prediction_{epoch}"
        predictions = np.load(result_dir / "predict_force_maps.npy")
        truths = np.load(result_dir / "truth_force_maps.npy")
        output_dir = result_dir

    regressions, plot_path, json_path = save_force_regression(
        predictions,
        truths,
        output_dir,
    )
    print(json.dumps(regressions, indent=2, allow_nan=True))
    print(f"Regression plot: {plot_path}")
    print(f"Regression values: {json_path}")


if __name__ == "__main__":
    main()
