"""
Shared I/O and tensor-shaping helpers for the ablation training pipeline.

Mirrors the equivalent helpers already used across scripts/data and
scripts/train in the main D-BRAN repo (build_multi_imu_input,
full_position_subset, rotmat_to_6d, angular_error_deg, ...), generalized
to an arbitrary number of local sensors per branch instead of the
five-branch design's fixed "root + one local sensor".
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F


def load_list(list_file: str) -> List[str]:
    path = Path(list_file)
    if not path.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")

    files = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing = [p for p in files if os.path.exists(p)]
    missing = len(files) - len(existing)
    if missing:
        print(f"[warning] Ignored {missing} missing paths from {list_file}")
    return existing


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def make_safe_name(src_path: str, suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(src_path))[0]
    digest = hashlib.md5(src_path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}_{suffix}.pt"


def ensure_acc_6x3(acceleration: torch.Tensor) -> torch.Tensor:
    if acceleration.dim() == 3 and tuple(acceleration.shape[1:]) == (6, 3):
        return acceleration.float().contiguous()
    if acceleration.dim() == 2 and acceleration.shape[1] == 18:
        return acceleration.reshape(-1, 6, 3).float().contiguous()
    raise ValueError(f"Unsupported acceleration shape: {tuple(acceleration.shape)}")


def ensure_ori_6x3x3(orientation: torch.Tensor) -> torch.Tensor:
    if orientation.dim() == 4 and tuple(orientation.shape[1:]) == (6, 3, 3):
        return orientation.float().contiguous()
    if orientation.dim() == 3 and tuple(orientation.shape[1:]) == (6, 9):
        return orientation.reshape(-1, 6, 3, 3).float().contiguous()
    if orientation.dim() == 2 and orientation.shape[1] == 54:
        return orientation.reshape(-1, 6, 3, 3).float().contiguous()
    raise ValueError(f"Unsupported orientation shape: {tuple(orientation.shape)}")


def build_multi_imu_input(
    acceleration: torch.Tensor,
    orientation: torch.Tensor,
    sensor_indices: Sequence[int],
) -> torch.Tensor:
    """
    [acc(3) + rotmat(9)] concatenated for each sensor in sensor_indices, in
    that order. For a branch with N sensors this is a 12*N-dim feature.
    """
    acceleration = ensure_acc_6x3(acceleration)
    orientation = ensure_ori_6x3x3(orientation)

    features = []
    for sensor_idx in sensor_indices:
        features.append(acceleration[:, sensor_idx, :])
        features.append(orientation[:, sensor_idx].reshape(orientation.shape[0], 9))

    return torch.cat(features, dim=1).float().contiguous()


def full_position_subset(
    full_position_69: torch.Tensor,
    joint_indices: Sequence[int],
) -> torch.Tensor:
    """
    Slice a [T, 69] (23 non-root joints x 3) position tensor down to the
    given joint indices (1..23), concatenated in the given order.
    """
    parts = []
    for joint_idx in joint_indices:
        if joint_idx == 0:
            raise ValueError("Position input does not include root joint 0.")
        start = (joint_idx - 1) * 3
        parts.append(full_position_69[:, start:start + 3])
    return torch.cat(parts, dim=1).float().contiguous()


def rotmat_to_6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    first_two_cols = rotation_matrix[..., :, :2]
    return (
        first_two_cols.transpose(-1, -2)
        .reshape(rotation_matrix.shape[:-2] + (6,))
        .float()
    )


def sixd_to_rotmat(x6d: torch.Tensor) -> torch.Tensor:
    x = x6d.reshape(*x6d.shape[:-1], 2, 3)
    a1 = x[..., 0, :]
    a2 = x[..., 1, :]

    b1 = F.normalize(a1, dim=-1)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def angular_error_deg(
    prediction_6d: torch.Tensor,
    target_6d: torch.Tensor,
    num_joints: int,
) -> torch.Tensor:
    prediction = prediction_6d.reshape(-1, num_joints, 6)
    target = target_6d.reshape(-1, num_joints, 6)

    prediction_matrix = sixd_to_rotmat(prediction)
    target_matrix = sixd_to_rotmat(target)

    relative = torch.matmul(prediction_matrix.transpose(-1, -2), target_matrix)
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    cosine = torch.clamp((trace - 1.0) / 2.0, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    return torch.rad2deg(torch.acos(cosine)).mean()
