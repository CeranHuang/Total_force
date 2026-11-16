"""Evaluation metrics and reporting helpers for node-force prediction."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


COMPONENT_NAMES = ("Fx", "Fy", "Fz")


class DistributionMetrics:
    """Accumulate node, contact, resultant, localization and no-load metrics."""

    def __init__(
        self,
        map_h=4,
        map_w=4,
        node_centers_xy=None,
        active_threshold_n=0.1,
        contact_threshold_n=0.05,
        tangent_angle_threshold_n=0.1,
        no_load_threshold_n=0.1,
        zero_epsilon_n=1e-6,
        active_node_force_std=None,
        total_force_std=None,
    ):
        self.map_h, self.map_w = int(map_h), int(map_w)
        self.active_threshold_n = float(active_threshold_n)
        self.contact_threshold_n = float(contact_threshold_n)
        self.tangent_angle_threshold_n = float(tangent_angle_threshold_n)
        self.no_load_threshold_n = float(no_load_threshold_n)
        self.zero_epsilon_n = float(zero_epsilon_n)
        self.active_node_force_std = (
            None
            if active_node_force_std is None
            else torch.as_tensor(active_node_force_std, dtype=torch.float64)
            .detach()
            .cpu()
            .reshape(3)
        )
        self.total_force_std = (
            None
            if total_force_std is None
            else torch.as_tensor(total_force_std, dtype=torch.float64)
            .detach()
            .cpu()
            .reshape(3)
        )
        if self.active_node_force_std is not None and (
            not torch.isfinite(self.active_node_force_std).all()
            or torch.any(self.active_node_force_std <= 0.0)
        ):
            raise ValueError(
                "active_node_force_std must contain three finite positive values."
            )
        if self.total_force_std is not None and (
            not torch.isfinite(self.total_force_std).all()
            or torch.any(self.total_force_std <= 0.0)
        ):
            raise ValueError("total_force_std must contain three finite positive values.")
        if node_centers_xy is None:
            yy, xx = torch.meshgrid(
                torch.arange(self.map_h, dtype=torch.float64),
                torch.arange(self.map_w, dtype=torch.float64),
                indexing="ij",
            )
            centers = torch.stack((xx, yy), dim=-1)
        else:
            centers = torch.as_tensor(node_centers_xy, dtype=torch.float64)
            if centers.shape == (self.map_h * self.map_w, 2):
                centers = centers.reshape(self.map_h, self.map_w, 2)
        if centers.shape != (self.map_h, self.map_w, 2):
            raise ValueError(f"Invalid node_centers_xy shape: {tuple(centers.shape)}")
        self.node_centers_xy = centers

        self.count = self.sample_count = 0
        self.abs_sum = torch.zeros(3, dtype=torch.float64)
        self.sq_sum = torch.zeros(3, dtype=torch.float64)
        self.per_node_abs_sum = torch.zeros(
            3, self.map_h, self.map_w, dtype=torch.float64
        )
        self.per_node_active_vector_abs_sum = torch.zeros(
            self.map_h, self.map_w, dtype=torch.float64
        )
        self.per_node_active_count = torch.zeros(
            self.map_h, self.map_w, dtype=torch.int64
        )
        self.vector_abs_sum = self.vector_sq_sum = 0.0
        self.active_count = 0
        self.active_abs_sum = torch.zeros(3, dtype=torch.float64)
        self.active_sq_sum = torch.zeros(3, dtype=torch.float64)
        self.tangent_magnitude_abs_sum = self.angle_sum = 0.0
        self.active_tangent_magnitude_abs_sum = 0.0
        self.tangent_angle_target_count = 0
        self.angle_count = 0
        self.total_tangent_magnitude_abs_sum = 0.0
        self.active_total_tangent_magnitude_abs_sum = 0.0
        self.total_tangent_angle_target_count = 0
        self.total_tangent_angles = []
        self.total_abs_sum = torch.zeros(3, dtype=torch.float64)
        self.total_sq_sum = torch.zeros(3, dtype=torch.float64)
        self.total_target_sum = torch.zeros(3, dtype=torch.float64)
        self.total_target_sq_sum = torch.zeros(3, dtype=torch.float64)
        self.total_magnitude_abs_sum = self.total_angle_sum = 0.0
        self.total_angle_count = 0
        self.contact_tp = self.contact_fp = self.contact_fn = 0
        self.cop_error_sum = 0.0
        self.cop_count = 0
        self.no_load_sample_count = self.no_load_node_count = 0
        self.no_load_abs_sum = torch.zeros(3, dtype=torch.float64)
        self.no_load_magnitude_sum = 0.0
        self.no_load_false_positive_nodes = 0
        self.no_load_false_positive_samples = 0

    def update(self, prediction, target):
        """Add a batch of matching ``[B, 3, H, W]`` force maps."""
        prediction = prediction.detach().cpu().double()
        target = target.detach().cpu().double()
        expected = (3, self.map_h, self.map_w)
        if tuple(prediction.shape[1:]) != expected or prediction.shape != target.shape:
            raise ValueError(
                f"Expected matching [B,3,{self.map_h},{self.map_w}] tensors, "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        batch_size = prediction.shape[0]
        error = prediction - target
        self.abs_sum += error.abs().sum(dim=(0, 2, 3))
        self.sq_sum += error.square().sum(dim=(0, 2, 3))
        self.per_node_abs_sum += error.abs().sum(dim=0)
        self.count += batch_size * self.map_h * self.map_w
        self.sample_count += batch_size

        vector_error = torch.linalg.vector_norm(error, dim=1)
        self.vector_abs_sum += vector_error.sum().item()
        self.vector_sq_sum += vector_error.square().sum().item()
        active = torch.linalg.vector_norm(target, dim=1) >= self.active_threshold_n
        active_count = int(active.sum().item())
        if active_count:
            for component in range(3):
                component_error = error[:, component][active]
                self.active_abs_sum[component] += component_error.abs().sum()
                self.active_sq_sum[component] += component_error.square().sum()
            self.per_node_active_vector_abs_sum += (vector_error * active).sum(dim=0)
            self.per_node_active_count += active.sum(dim=0)
            self.active_count += active_count

        pred_t, target_t = prediction[:, :2], target[:, :2]
        pred_mag = torch.linalg.vector_norm(pred_t, dim=1)
        target_mag = torch.linalg.vector_norm(target_t, dim=1)
        tangent_magnitude_error = (
            pred_mag - target_mag
        ).abs()
        self.tangent_magnitude_abs_sum += tangent_magnitude_error.sum().item()
        target_valid = target_mag >= self.tangent_angle_threshold_n
        self.tangent_angle_target_count += int(target_valid.sum().item())
        if target_valid.any():
            self.active_tangent_magnitude_abs_sum += (
                tangent_magnitude_error[target_valid].sum().item()
            )
        angle_valid = target_valid & (
            pred_mag >= self.tangent_angle_threshold_n
        )
        if angle_valid.any():
            cosine = (pred_t * target_t).sum(dim=1) / (
                pred_mag * target_mag
            ).clamp_min(1e-8)
            angles = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
            self.angle_sum += angles[angle_valid].sum().item()
            self.angle_count += int(angle_valid.sum().item())

        pred_total = prediction.sum(dim=(2, 3))
        target_total = target.sum(dim=(2, 3))
        pred_total_t = pred_total[:, :2]
        target_total_t = target_total[:, :2]
        pred_total_t_mag = torch.linalg.vector_norm(pred_total_t, dim=1)
        target_total_t_mag = torch.linalg.vector_norm(target_total_t, dim=1)
        total_tangent_magnitude_error = (
            pred_total_t_mag - target_total_t_mag
        ).abs()
        self.total_tangent_magnitude_abs_sum += (
            total_tangent_magnitude_error.sum().item()
        )
        total_tangent_target_valid = (
            target_total_t_mag >= self.tangent_angle_threshold_n
        )
        self.total_tangent_angle_target_count += int(
            total_tangent_target_valid.sum().item()
        )
        if total_tangent_target_valid.any():
            self.active_total_tangent_magnitude_abs_sum += (
                total_tangent_magnitude_error[
                    total_tangent_target_valid
                ].sum().item()
            )
        total_tangent_valid = total_tangent_target_valid & (
            pred_total_t_mag >= self.tangent_angle_threshold_n
        )
        if total_tangent_valid.any():
            cosine = (pred_total_t * target_total_t).sum(dim=1) / (
                pred_total_t_mag * target_total_t_mag
            ).clamp_min(1e-8)
            angles = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
            self.total_tangent_angles.extend(
                angles[total_tangent_valid].tolist()
            )
        total_error = pred_total - target_total
        self.total_abs_sum += total_error.abs().sum(dim=0)
        self.total_sq_sum += total_error.square().sum(dim=0)
        self.total_target_sum += target_total.sum(dim=0)
        self.total_target_sq_sum += target_total.square().sum(dim=0)
        pred_total_mag = torch.linalg.vector_norm(pred_total, dim=1)
        target_total_mag = torch.linalg.vector_norm(target_total, dim=1)
        self.total_magnitude_abs_sum += (
            pred_total_mag - target_total_mag
        ).abs().sum().item()
        total_valid = target_total_mag >= self.active_threshold_n
        if total_valid.any():
            cosine = (pred_total * target_total).sum(dim=1) / (
                pred_total_mag * target_total_mag
            ).clamp_min(1e-8)
            angles = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
            self.total_angle_sum += angles[total_valid].sum().item()
            self.total_angle_count += int(total_valid.sum().item())

        # Compression is negative Fz. Positive Fz is tension and does not
        # count as contact or contribute to the center of pressure.
        pred_normal = (-prediction[:, 2]).clamp_min(0.0)
        target_normal = (-target[:, 2]).clamp_min(0.0)
        pred_contact = pred_normal >= self.contact_threshold_n
        target_contact = target_normal >= self.contact_threshold_n
        self.contact_tp += int((pred_contact & target_contact).sum().item())
        self.contact_fp += int((pred_contact & ~target_contact).sum().item())
        self.contact_fn += int((~pred_contact & target_contact).sum().item())

        pred_weight_sum = pred_normal.sum(dim=(1, 2))
        target_weight_sum = target_normal.sum(dim=(1, 2))
        cop_valid = (
            (pred_weight_sum >= self.contact_threshold_n)
            & (target_weight_sum >= self.contact_threshold_n)
        )
        if cop_valid.any():
            centers = self.node_centers_xy
            pred_cop = (
                (pred_normal[..., None] * centers).sum(dim=(1, 2))
                / pred_weight_sum[:, None].clamp_min(1e-8)
            )
            target_cop = (
                (target_normal[..., None] * centers).sum(dim=(1, 2))
                / target_weight_sum[:, None].clamp_min(1e-8)
            )
            cop_error = torch.linalg.vector_norm(pred_cop - target_cop, dim=1)
            self.cop_error_sum += cop_error[cop_valid].sum().item()
            self.cop_count += int(cop_valid.sum().item())

        no_load = target.abs().amax(dim=(1, 2, 3)) <= self.zero_epsilon_n
        if no_load.any():
            no_load_prediction = prediction[no_load]
            no_load_magnitude = torch.linalg.vector_norm(no_load_prediction, dim=1)
            self.no_load_sample_count += int(no_load.sum().item())
            self.no_load_node_count += int(no_load_magnitude.numel())
            self.no_load_abs_sum += no_load_prediction.abs().sum(dim=(0, 2, 3))
            self.no_load_magnitude_sum += no_load_magnitude.sum().item()
            false_positive = no_load_magnitude >= self.no_load_threshold_n
            self.no_load_false_positive_nodes += int(false_positive.sum().item())
            self.no_load_false_positive_samples += int(
                false_positive.flatten(1).any(dim=1).sum().item()
            )

    def compute(self):
        """Return all accumulated metrics as tensors or Python scalars."""
        if self.count == 0:
            raise ValueError("No samples were available for metric calculation.")
        nan3 = torch.full((3,), float("nan"), dtype=torch.float32)
        if self.active_count:
            sparse_mae = (self.active_abs_sum / self.active_count).float()
            sparse_rmse = torch.sqrt(
                self.active_sq_sum / self.active_count
            ).float()
        else:
            sparse_mae, sparse_rmse = nan3.clone(), nan3.clone()
        active_node_nmae = (
            sparse_mae.double() / self.active_node_force_std
            if self.active_node_force_std is not None
            else torch.full((3,), float("nan"), dtype=torch.float64)
        )
        active_per_node_vector_mae = (
            self.per_node_active_vector_abs_sum
            / self.per_node_active_count.clamp_min(1)
        ).float()
        active_per_node_vector_mae[self.per_node_active_count == 0] = float("nan")
        if self.total_tangent_angles:
            total_tangent_angles = torch.tensor(
                self.total_tangent_angles, dtype=torch.float64
            )
            total_tangent_angle_mean = total_tangent_angles.mean().item()
            total_tangent_angle_median = torch.quantile(
                total_tangent_angles, 0.5
            ).item()
            total_tangent_angle_p90 = torch.quantile(
                total_tangent_angles, 0.9
            ).item()
        else:
            total_tangent_angle_mean = float("nan")
            total_tangent_angle_median = float("nan")
            total_tangent_angle_p90 = float("nan")
        precision_den = self.contact_tp + self.contact_fp
        recall_den = self.contact_tp + self.contact_fn
        precision = self.contact_tp / precision_den if precision_den else 0.0
        recall = self.contact_tp / recall_den if recall_den else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        iou_den = self.contact_tp + self.contact_fp + self.contact_fn
        iou = self.contact_tp / iou_den if iou_den else 1.0
        total_force_mae = self.total_abs_sum / self.sample_count
        total_target_ss = self.total_target_sq_sum - (
            self.total_target_sum.square() / self.sample_count
        )
        total_force_r2 = torch.full((3,), float("nan"), dtype=torch.float64)
        varying = total_target_ss > 0.0
        total_force_r2[varying] = 1.0 - self.total_sq_sum[varying] / total_target_ss[varying]
        total_force_nmae = (
            total_force_mae / self.total_force_std
            if self.total_force_std is not None
            else torch.full((3,), float("nan"), dtype=torch.float64)
        )
        return {
            "mae": (self.abs_sum / self.count).float(),
            "rmse": torch.sqrt(self.sq_sum / self.count).float(),
            "per_node_mae": (self.per_node_abs_sum / self.sample_count).float(),
            "active_per_node_vector_mae": active_per_node_vector_mae,
            "active_per_node_count": self.per_node_active_count.clone(),
            "node_vector_mae": self.vector_abs_sum / self.count,
            "node_vector_rmse": (self.vector_sq_sum / self.count) ** 0.5,
            "sparse_mae": sparse_mae,
            "sparse_rmse": sparse_rmse,
            "active_node_mae": sparse_mae,
            "active_node_nmae": active_node_nmae.float(),
            "active_node_rmse": sparse_rmse,
            "active_node_count": self.active_count,
            "active_node_fraction": self.active_count / self.count,
            "tangent_magnitude_mae": self.tangent_magnitude_abs_sum / self.count,
            "active_tangent_magnitude_mae": (
                self.active_tangent_magnitude_abs_sum
                / self.tangent_angle_target_count
                if self.tangent_angle_target_count
                else float("nan")
            ),
            "tangent_angle_deg": (
                self.angle_sum / self.angle_count
                if self.angle_count
                else float("nan")
            ),
            "tangent_angle_count": self.angle_count,
            "tangent_angle_valid_count": self.angle_count,
            "tangent_angle_target_count": self.tangent_angle_target_count,
            "tangent_angle_coverage": (
                self.angle_count / self.tangent_angle_target_count
                if self.tangent_angle_target_count
                else float("nan")
            ),
            "total_tangent_magnitude_mae": (
                self.total_tangent_magnitude_abs_sum / self.sample_count
            ),
            "active_total_tangent_magnitude_mae": (
                self.active_total_tangent_magnitude_abs_sum
                / self.total_tangent_angle_target_count
                if self.total_tangent_angle_target_count
                else float("nan")
            ),
            "total_tangent_angle_deg": total_tangent_angle_mean,
            "total_tangent_angle_median_deg": total_tangent_angle_median,
            "total_tangent_angle_p90_deg": total_tangent_angle_p90,
            "total_tangent_angle_count": len(self.total_tangent_angles),
            "total_tangent_angle_valid_count": len(self.total_tangent_angles),
            "total_tangent_angle_target_count": (
                self.total_tangent_angle_target_count
            ),
            "total_tangent_angle_coverage": (
                len(self.total_tangent_angles)
                / self.total_tangent_angle_target_count
                if self.total_tangent_angle_target_count
                else float("nan")
            ),
            "total_force_mae": total_force_mae.float(),
            "total_force_rmse": torch.sqrt(
                self.total_sq_sum / self.sample_count
            ).float(),
            "total_force_nmae": total_force_nmae.float(),
            "total_force_r2": total_force_r2.float(),
            "total_magnitude_mae": (
                self.total_magnitude_abs_sum / self.sample_count
            ),
            "total_angle_deg": (
                self.total_angle_sum / self.total_angle_count
                if self.total_angle_count
                else float("nan")
            ),
            "contact_precision": precision,
            "contact_recall": recall,
            "contact_f1": f1,
            "contact_iou": iou,
            "cop_error_mm": (
                self.cop_error_sum / self.cop_count
                if self.cop_count
                else float("nan")
            ),
            "cop_count": self.cop_count,
            "no_load_sample_count": self.no_load_sample_count,
            "no_load_component_mae": (
                (self.no_load_abs_sum / self.no_load_node_count).float()
                if self.no_load_node_count
                else nan3.clone()
            ),
            "no_load_node_magnitude_mean": (
                self.no_load_magnitude_sum / self.no_load_node_count
                if self.no_load_node_count
                else float("nan")
            ),
            "no_load_node_fpr": (
                self.no_load_false_positive_nodes / self.no_load_node_count
                if self.no_load_node_count
                else float("nan")
            ),
            "no_load_sample_fpr": (
                self.no_load_false_positive_samples / self.no_load_sample_count
                if self.no_load_sample_count
                else float("nan")
            ),
        }


def format_metrics(metrics):
    """Format primary active-node metrics and secondary global metrics."""
    all_node_mae = metrics["mae"].detach().cpu().numpy()
    active_node_mae = metrics["active_node_mae"].detach().cpu().numpy()
    active_node_nmae = metrics["active_node_nmae"].detach().cpu().numpy()
    active_node_rmse = metrics["active_node_rmse"].detach().cpu().numpy()
    total_mae = metrics["total_force_mae"].detach().cpu().numpy()
    total_nmae = metrics["total_force_nmae"].detach().cpu().numpy()
    total_r2 = metrics["total_force_r2"].detach().cpu().numpy()
    return (
        f"Active-node MAE={np.round(active_node_mae, 5)}, "
        f"NMAE={np.round(active_node_nmae, 5)}, "
        f"RMSE={np.round(active_node_rmse, 5)} "
        f"(n={metrics['active_node_count']}, "
        f"{metrics['active_node_fraction']:.1%} of nodes); "
        f"All-node MAE={np.round(all_node_mae, 5)} [secondary]; "
        f"Total-force MAE={np.round(total_mae, 5)}, "
        f"NMAE={np.round(total_nmae, 5)}, R2={np.round(total_r2, 5)}, "
        f"magnitude MAE={metrics['total_magnitude_mae']:.5f}; "
        f"T-mag MAE active-node/total={metrics['active_tangent_magnitude_mae']:.5f}/"
        f"{metrics['active_total_tangent_magnitude_mae']:.5f}; "
        f"Total T-angle mean/median/P90="
        f"{metrics['total_tangent_angle_deg']:.2f}/"
        f"{metrics['total_tangent_angle_median_deg']:.2f}/"
        f"{metrics['total_tangent_angle_p90_deg']:.2f} deg "
        f"(coverage={metrics['total_tangent_angle_coverage']:.1%}, "
        f"n={metrics['total_tangent_angle_valid_count']}/"
        f"{metrics['total_tangent_angle_target_count']}); "
        f"Contact P/R/F1/IoU={metrics['contact_precision']:.3f}/"
        f"{metrics['contact_recall']:.3f}/{metrics['contact_f1']:.3f}/"
        f"{metrics['contact_iou']:.3f}; CoP={metrics['cop_error_mm']:.3f} mm; "
        f"No-load node/sample FPR={metrics['no_load_node_fpr']:.3f}/"
        f"{metrics['no_load_sample_fpr']:.3f}"
    )

def serializable_metrics(metrics):
    """Convert metric tensors and NumPy scalars to JSON-compatible values."""
    result = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def save_evaluation_visualizations(result_dir, predictions, truths, metrics):
    """Save active-node, total-force and operating-point evaluation figures."""
    result_dir = Path(result_dir)
    all_predictions = (
        np.concatenate(predictions, axis=0)
        if isinstance(predictions, (list, tuple))
        else np.asarray(predictions)
    )
    all_truths = (
        np.concatenate(truths, axis=0)
        if isinstance(truths, (list, tuple))
        else np.asarray(truths)
    )
    predicted_totals = all_predictions.sum(axis=(1, 2))
    true_totals = all_truths.sum(axis=(1, 2))
    component_labels = list(COMPONENT_NAMES)
    active_mae = np.asarray(metrics["active_node_mae"], dtype=float)
    active_rmse = np.asarray(metrics["active_node_rmse"], dtype=float)
    total_mae = np.asarray(metrics["total_force_mae"], dtype=float)
    total_rmse = np.asarray(metrics["total_force_rmse"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    x = np.arange(len(component_labels))
    width = 0.36
    axes[0, 0].bar(x - width / 2, active_mae, width, label="MAE")
    axes[0, 0].bar(x + width / 2, active_rmse, width, label="RMSE")
    axes[0, 0].set_xticks(x, component_labels)
    axes[0, 0].set_title("Active-node component error")
    axes[0, 0].set_ylabel("Force (N)")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(x - width / 2, total_mae, width, label="MAE")
    axes[0, 1].bar(x + width / 2, total_rmse, width, label="RMSE")
    axes[0, 1].set_xticks(x, component_labels)
    axes[0, 1].set_title("Per-sample total-force component error")
    axes[0, 1].set_ylabel("Force (N)")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    contact_names = ["Precision", "Recall", "F1", "IoU"]
    contact_values = [
        metrics["contact_precision"],
        metrics["contact_recall"],
        metrics["contact_f1"],
        metrics["contact_iou"],
    ]
    axes[1, 0].bar(contact_names, contact_values, color="tab:green")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Contact detection")
    axes[1, 0].grid(axis="y", alpha=0.3)

    active_per_node_mae = np.asarray(
        metrics["active_per_node_vector_mae"], dtype=float
    )
    image = axes[1, 1].imshow(active_per_node_mae, cmap="magma")
    axes[1, 1].set_title("Active-node vector MAE")
    axes[1, 1].set_xlabel("Node column")
    axes[1, 1].set_ylabel("Node row")
    axes[1, 1].set_xticks(range(active_per_node_mae.shape[1]))
    axes[1, 1].set_yticks(range(active_per_node_mae.shape[0]))
    fig.colorbar(image, ax=axes[1, 1], label="Force (N)")
    fig.savefig(result_dir / "evaluation_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    def finite_value(value):
        value = float(value)
        return value if np.isfinite(value) else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    tangent_names = ["Active node", "Active sample total"]
    tangent_values = [
        finite_value(metrics["active_tangent_magnitude_mae"]),
        finite_value(metrics["active_total_tangent_magnitude_mae"]),
    ]
    axes[0, 0].bar(tangent_names, tangent_values, color="tab:blue")
    axes[0, 0].set_title("Tangential magnitude MAE")
    axes[0, 0].set_ylabel("Force (N)")
    axes[0, 0].grid(axis="y", alpha=0.3)

    angle_names = ["Mean", "Median", "P90"]
    angle_values = [
        finite_value(metrics["total_tangent_angle_deg"]),
        finite_value(metrics["total_tangent_angle_median_deg"]),
        finite_value(metrics["total_tangent_angle_p90_deg"]),
    ]
    axes[0, 1].bar(angle_names, angle_values, color="tab:orange")
    axes[0, 1].set_title("Total tangential direction error")
    axes[0, 1].set_ylabel("Angle (deg)")
    axes[0, 1].grid(axis="y", alpha=0.3)

    no_load_names = ["Node FPR", "Sample FPR"]
    no_load_values = [
        finite_value(metrics["no_load_node_fpr"]),
        finite_value(metrics["no_load_sample_fpr"]),
    ]
    axes[1, 0].bar(no_load_names, no_load_values, color="tab:red")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("No-load false-positive rate")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.75,
        f"Total magnitude MAE: {metrics['total_magnitude_mae']:.3f} N\n"
        f"CoP error: {metrics['cop_error_mm']:.3f} mm\n"
        f"Active nodes: {metrics['active_node_count']} "
        f"({metrics['active_node_fraction']:.1%})\n"
        f"Node tangent angle coverage: {metrics['tangent_angle_coverage']:.1%} "
        f"({metrics['tangent_angle_valid_count']}/"
        f"{metrics['tangent_angle_target_count']})\n"
        f"Total tangent angle coverage: "
        f"{metrics['total_tangent_angle_coverage']:.1%} "
        f"({metrics['total_tangent_angle_valid_count']}/"
        f"{metrics['total_tangent_angle_target_count']})",
        va="top",
        fontsize=12,
    )
    axes[1, 1].set_title("Resultant / localization summary")
    fig.savefig(
        result_dir / "tangent_operating_metrics.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for index, (ax, component) in enumerate(zip(axes, COMPONENT_NAMES)):
        truth_axis = true_totals[:, index]
        prediction_axis = predicted_totals[:, index]
        low = min(float(truth_axis.min()), float(prediction_axis.min()))
        high = max(float(truth_axis.max()), float(prediction_axis.max()))
        if low == high:
            low, high = low - 0.5, high + 0.5
        ax.scatter(truth_axis, prediction_axis, s=12, alpha=0.6)
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set_title(
            f"{component}: NMAE={metrics['total_force_nmae'][index]:.3f}, "
            f"R2={metrics['total_force_r2'][index]:.3f}"
        )
        ax.set_xlabel("Ground truth (N)")
        ax.set_ylabel("Prediction (N)")
        ax.grid(alpha=0.3)
    fig.savefig(result_dir / "force_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    residuals = predicted_totals - true_totals
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for index, (ax, component) in enumerate(zip(axes, COMPONENT_NAMES)):
        ax.scatter(true_totals[:, index], residuals[:, index], s=12, alpha=0.6)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(component)
        ax.set_xlabel("Ground truth (N)")
        ax.set_ylabel("Residual: prediction - truth (N)")
        ax.grid(alpha=0.3)
    fig.savefig(result_dir / "force_residuals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    truth = np.asarray(all_truths[0], dtype=float)
    prediction = np.asarray(all_predictions[0], dtype=float)
    error = prediction - truth
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    titles = ("Truth", "Prediction", "Error")
    for row, component in enumerate(COMPONENT_NAMES):
        limit = max(
            float(np.max(np.abs(truth[:, :, row]))),
            float(np.max(np.abs(prediction[:, :, row]))),
            1e-6,
        )
        for col, (data, title) in enumerate(
            zip(
                (truth[:, :, row], prediction[:, :, row], error[:, :, row]),
                titles,
            )
        ):
            image = axes[row, col].imshow(
                data, cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axes[row, col].set_title(f"{component} - {title}")
            axes[row, col].set_xticks(range(truth.shape[1]))
            axes[row, col].set_yticks(range(truth.shape[0]))
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.savefig(
        result_dir / "force_map_comparison.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)
