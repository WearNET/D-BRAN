"""
Prepare Pose-S3 training-ready tensors for an ablation variant.

Per branch:
    input  = IMU(root + local sensors) + Pose-S2 predicted position
             subset for that branch's position_joints
    target = root-relative global 6D rotation, sliced to that branch's
             reduced_joints (recomputed from raw pose here, same as
             scripts/data/prepare_pose_s3_five_branch.py -- this rotation
             target is branch-agnostic and does not depend on any
             checkpoint, only Pose-S2's POSITION input does).
"""

from __future__ import annotations

# BEGIN D-BRAN PROJECT BOOTSTRAP
import sys as _dbran_sys
from pathlib import Path as _DbranPath

_dbran_current_file = _DbranPath(__file__).resolve()

for _dbran_candidate in (_dbran_current_file.parent, *_dbran_current_file.parents):
    if (_dbran_candidate / "main_path.py").is_file():
        _dbran_root_string = str(_dbran_candidate)
        if _dbran_root_string not in _dbran_sys.path:
            _dbran_sys.path.insert(0, _dbran_root_string)
        break
else:
    raise RuntimeError(f"Could not locate the D-BRAN project root from {_dbran_current_file}")

from main_path import PROJECT_ROOT
# END D-BRAN PROJECT BOOTSTRAP
import articulate as art
from config import paths

_ablation_root = _dbran_current_file.parent.parent
if str(_ablation_root) not in _dbran_sys.path:
    _dbran_sys.path.insert(0, str(_ablation_root))

from common.branch_configs import REDUCED_JOINTS, get_variant, validate_variant
from common.io_utils import (
    build_multi_imu_input,
    full_position_subset,
    load_list,
    normalize_path,
    rotmat_to_6d,
)


import argparse
import hashlib
import os
from pathlib import Path

import torch
from tqdm import tqdm

ROOT_SENSOR_IDX = 5

POSE_S2_POSITION_KEYS = ["full_joint_position_pred", "full_joint_position_gt"]


def build_source_map(files, label):
    mapping = {}
    for path in tqdm(files, desc=f"Indexing {label}", ncols=120):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "source_file" not in data:
            raise KeyError(f"'source_file' missing in {path}")
        source = normalize_path(str(data["source_file"]))
        if source in mapping:
            raise RuntimeError(f"Duplicate source_file in {label}: {source}")
        mapping[source] = path
    return mapping


def get_pose_s2_full_position(pose_s2_data):
    for key in POSE_S2_POSITION_KEYS:
        value = pose_s2_data.get(key)
        if isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[1] == 69:
            return value.float()
    raise KeyError(f"Could not find [T,69] Pose-S2 position. Keys: {list(pose_s2_data.keys())}")


def compute_reduced_root_relative_global_6d(raw_data, body_model):
    pose = raw_data["pose"].float()
    if pose.dim() == 2 and pose.shape[1] == 72:
        pose = pose.reshape(pose.shape[0], 24, 3)

    pose_rotmat = art.math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
    global_rotmat, _ = body_model.forward_kinematics(pose_rotmat, calc_mesh=False)

    root_rotmat = global_rotmat[:, 0].unsqueeze(1)
    root_inverse = root_rotmat.transpose(-1, -2)
    root_relative_global = torch.matmul(root_inverse, global_rotmat)

    reduced = root_relative_global[:, REDUCED_JOINTS]
    return rotmat_to_6d(reduced).reshape(reduced.shape[0], len(REDUCED_JOINTS) * 6).float().contiguous()


def target_branch_from_reduced(target_reduced_6d, reduced_joints):
    target = target_reduced_6d.reshape(target_reduced_6d.shape[0], len(REDUCED_JOINTS), 6)
    parts = [target[:, REDUCED_JOINTS.index(j), :] for j in reduced_joints]
    return torch.cat(parts, dim=1).float().contiguous()


def make_output_name(source_file: str) -> str:
    normalized = normalize_path(source_file)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    stem = Path(source_file).stem
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{safe_stem}_{digest}_pose_s3_data.pt"


def process_one(raw_path, pose_s2_path, body_model, variant, output_dir):
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    pose_s2_data = torch.load(pose_s2_path, map_location="cpu", weights_only=False)

    pose_s2_full = get_pose_s2_full_position(pose_s2_data)
    target_reduced_6d = compute_reduced_root_relative_global_6d(raw, body_model)

    sequence_length = min(raw["acc"].shape[0], raw["ori"].shape[0], pose_s2_full.shape[0], target_reduced_6d.shape[0])

    item = {
        "source_file": raw_path,
        "pose_s2_source_file": pose_s2_path,
        "num_frames": sequence_length,
        "pose_s2_full_position_input": pose_s2_full[:sequence_length],
        "pose3_target_6d_reduced": target_reduced_6d[:sequence_length],
        "branch_order": variant["order"],
    }

    for branch in variant["order"]:
        config = variant["s3"][branch]
        imu_features = build_multi_imu_input(raw["acc"], raw["ori"], config["sensor_indices"])[:sequence_length]
        position_features = full_position_subset(pose_s2_full, config["position_joints"])[:sequence_length]
        branch_input = torch.cat([imu_features, position_features], dim=1).float().contiguous()
        branch_target = target_branch_from_reduced(target_reduced_6d, config["reduced_joints"])[:sequence_length]

        if branch_input.shape[1] != config["input_dim"]:
            raise RuntimeError(f"{branch}_input expected dim {config['input_dim']}, got {branch_input.shape[1]}")
        if branch_target.shape[1] != config["output_dim"]:
            raise RuntimeError(f"{branch}_target expected dim {config['output_dim']}, got {branch_target.shape[1]}")

        item[f"{branch}_input"] = branch_input
        item[f"{branch}_target_6d_reduced"] = branch_target

    output_path = os.path.join(output_dir, make_output_name(raw_path))
    torch.save(item, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--raw_list_file", required=True)
    parser.add_argument("--pose_s2_list_file", required=True, help="Output of precompute_pose_s2_predictions.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    parser.add_argument("--skip_errors", action="store_true")
    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)

    os.makedirs(args.output_dir, exist_ok=True)

    raw_files = load_list(args.raw_list_file)
    pose_s2_files = load_list(args.pose_s2_list_file)
    pose_s2_map = build_source_map(pose_s2_files, "Pose-S2 outputs")

    body_model = art.ParametricModel(paths.smpl_file, device=torch.device("cpu"))

    saved_paths, skipped = [], []
    for raw_path in tqdm(raw_files, desc=f"Preparing Pose-S3 data ({args.variant})", ncols=120):
        source = normalize_path(raw_path)
        if source not in pose_s2_map:
            skipped.append((raw_path, "missing Pose-S2 output"))
            continue
        try:
            saved_paths.append(process_one(raw_path, pose_s2_map[source], body_model, variant, args.output_dir))
        except Exception as error:
            if not args.skip_errors:
                raise
            skipped.append((raw_path, repr(error)))

    output_parent = os.path.dirname(args.output_list_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(args.output_list_file, "w", encoding="utf-8") as f:
        for path in saved_paths:
            f.write(path + "\n")

    print("\nDone.")
    print("  Raw files:", len(raw_files))
    print("  Saved:", len(saved_paths))
    print("  Skipped:", len(skipped))
    if skipped:
        print("  Skipped examples:", skipped[:10])


if __name__ == "__main__":
    main()
