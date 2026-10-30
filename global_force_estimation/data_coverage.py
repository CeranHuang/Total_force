"""Audit node-force labels for long-term multi-contact training coverage."""

from __future__ import annotations

import math
from collections import Counter

import torch


def _as_force_map(labels, map_h, map_w):
    labels = torch.as_tensor(labels, dtype=torch.float32)
    expected = 3 * map_h * map_w
    if labels.ndim == 2 and tuple(labels.shape) == (map_h * map_w, 3):
        labels = labels.reshape(1, map_h, map_w, 3)
    elif labels.ndim == 2 and labels.shape[1] == expected:
        labels = labels.reshape(-1, map_h, map_w, 3)
    elif labels.ndim == 3 and tuple(labels.shape) in (
        (map_h, map_w, 3),
        (3, map_h, map_w),
    ):
        labels = labels.unsqueeze(0)
    if labels.ndim != 4:
        raise ValueError(f"Unsupported force-label shape: {tuple(labels.shape)}.")
    if tuple(labels.shape[1:]) == (map_h, map_w, 3):
        labels = labels.permute(0, 3, 1, 2)
    if tuple(labels.shape[1:]) != (3, map_h, map_w):
        raise ValueError(f"Unsupported force-label shape: {tuple(labels.shape)}.")
    return labels.contiguous()


def analyze_force_label_coverage(
    loader,
    map_h=4,
    map_w=4,
    contact_threshold_n=0.05,
    tangent_threshold_n=0.1,
    direction_bins=8,
    multi_direction_separation_deg=30.0,
):
    """Return JSON-serializable coverage statistics for a force-label loader.

    Active nodes use compressive normal force max(-Fz, 0). A sample is called
    multi-direction only when at least two active tangential vectors exceed
    tangent_threshold_n and differ by the configured angle.
    """
    contact_threshold_n = float(contact_threshold_n)
    tangent_threshold_n = float(tangent_threshold_n)
    direction_bins = int(direction_bins)
    separation_rad = math.radians(float(multi_direction_separation_deg))
    if min(contact_threshold_n, tangent_threshold_n) < 0.0:
        raise ValueError("Coverage thresholds must be non-negative.")
    if direction_bins < 1:
        raise ValueError("direction_bins must be positive.")

    sample_count = 0
    active_histogram = Counter()
    node_active_counts = torch.zeros(map_h, map_w, dtype=torch.int64)
    direction_counts = torch.zeros(direction_bins, dtype=torch.int64)
    multi_direction_samples = 0
    total_force_min = torch.full((3,), float("inf"), dtype=torch.float64)
    total_force_max = torch.full((3,), float("-inf"), dtype=torch.float64)

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Coverage loader must return (image, label[, path]).")
        force_map = _as_force_map(batch[1], map_h, map_w).double()
        active = (-force_map[:, 2]).clamp_min(0.0) >= contact_threshold_n
        active_counts = active.flatten(1).sum(dim=1)
        node_active_counts += active.sum(dim=0).cpu()
        total_force = force_map.sum(dim=(2, 3))
        total_force_min = torch.minimum(total_force_min, total_force.min(dim=0).values)
        total_force_max = torch.maximum(total_force_max, total_force.max(dim=0).values)

        tangent = force_map[:, :2].permute(0, 2, 3, 1)
        tangent_magnitude = torch.linalg.vector_norm(tangent, dim=-1)
        direction_valid = active & (tangent_magnitude >= tangent_threshold_n)
        angles = torch.atan2(tangent[..., 1], tangent[..., 0])
        bins = torch.floor((angles + math.pi) / (2.0 * math.pi) * direction_bins)
        bins = bins.long().clamp(0, direction_bins - 1)
        for index in range(force_map.shape[0]):
            count = int(active_counts[index].item())
            active_histogram[count] += 1
            valid_angles = angles[index][direction_valid[index]]
            if valid_angles.numel():
                direction_counts += torch.bincount(
                    bins[index][direction_valid[index]].cpu(),
                    minlength=direction_bins,
                )
            if valid_angles.numel() >= 2:
                delta = torch.abs(valid_angles[:, None] - valid_angles[None, :])
                delta = torch.minimum(delta, 2.0 * math.pi - delta)
                if torch.any(delta >= separation_rad):
                    multi_direction_samples += 1
        sample_count += force_map.shape[0]

    if sample_count == 0:
        raise ValueError("Cannot audit an empty dataset.")

    maximum_nodes = map_h * map_w
    histogram = [int(active_histogram.get(index, 0)) for index in range(maximum_nodes + 1)]
    two_plus = int(sum(histogram[2:]))
    three_plus = int(sum(histogram[3:]))
    occupied_nodes = int((node_active_counts > 0).sum().item())
    covered_direction_bins = int((direction_counts > 0).sum().item())
    warnings = []
    if histogram[0] == 0:
        warnings.append("No no-load labels were found.")
    if two_plus == 0:
        warnings.append("No labels contain two or more active nodes.")
    if three_plus == 0:
        warnings.append("No labels contain three or more active nodes.")
    if multi_direction_samples == 0:
        warnings.append("No sample contains two verified, different tangential directions.")
    if occupied_nodes < maximum_nodes:
        warnings.append(
            f"Only {occupied_nodes}/{maximum_nodes} node positions appear as active."
        )
    if covered_direction_bins < direction_bins:
        warnings.append(
            f"Tangential directions cover {covered_direction_bins}/{direction_bins} bins."
        )

    return {
        "sample_count": int(sample_count),
        "thresholds_n": {
            "contact": contact_threshold_n,
            "tangent_direction": tangent_threshold_n,
        },
        "active_node_count_histogram": {
            str(index): count for index, count in enumerate(histogram) if count
        },
        "no_load_samples": histogram[0],
        "single_active_node_samples": histogram[1],
        "two_or_more_active_node_samples": two_plus,
        "three_or_more_active_node_samples": three_plus,
        "maximum_active_nodes": max(active_histogram),
        "multi_direction_samples": int(multi_direction_samples),
        "node_active_counts": node_active_counts.tolist(),
        "occupied_node_positions": occupied_nodes,
        "direction_bin_counts": direction_counts.tolist(),
        "covered_direction_bins": covered_direction_bins,
        "total_force_min_n": total_force_min.tolist(),
        "total_force_max_n": total_force_max.tolist(),
        "interpretation": (
            "Multiple active labels may come from geometric force sharing; this audit "
            "cannot prove that the sample was collected with simultaneous physical contacts."
        ),
        "warnings": warnings,
    }

