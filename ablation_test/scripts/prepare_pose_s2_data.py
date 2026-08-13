"""
Prepare Pose-S2 training-ready tensors for an ablation variant.

Per branch:
    input  = IMU(root + local sensors) + that branch's Pose-S1 predicted
             leaf position(s)
    target = full_joint_position_gt (from the existing, branch-agnostic
             pose_s2_gt_train/test files) sliced to that branch's joints

Mirrors scripts/data/prepare_pose_s2_full_distributed.py, generalized to
N local sensors per branch instead of a fixed root+1-local design.
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

_ablation_root = _dbran_current_file.parent.parent
if str(_ablation_root) not in _dbran_sys.path:
    _dbran_sys.path.insert(0, str(_ablation_root))

from common.branch_configs import get_variant, validate_variant
from common.io_utils import (
    build_multi_imu_input,
    full_position_subset,
    load_list,
    normalize_path,
)


import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

ROOT_SENSOR_IDX = 5

POSE_S2_GT_KEYS = ["full_joint_position_gt", "p_all_gt_flat"]


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


def get_full_position_gt(data):
    for key in POSE_S2_GT_KEYS:
        value = data.get(key)
        if isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[1] == 69:
            return value.float()
    raise KeyError(f"Could not find a [T,69] Pose-S2 GT tensor. Keys: {list(data.keys())}")


def make_output_name(source_file: str) -> str:
    import hashlib
    normalized = normalize_path(source_file)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    stem = Path(source_file).stem
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{safe_stem}_{digest}_pose_s2_data.pt"


def process_one(raw_path, s1_pred_path, gt_path, variant, output_dir):
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    s1_data = torch.load(s1_pred_path, map_location="cpu", weights_only=False)
    gt_data = torch.load(gt_path, map_location="cpu", weights_only=False)

    full_position_gt = get_full_position_gt(gt_data)

    item = {
        "source_file": raw_path,
        "s1_pred_source_file": s1_pred_path,
        "pose_s2_gt_source_file": gt_path,
        "full_joint_position_gt": full_position_gt,
        "branch_order": variant["order"],
    }

    lengths = [full_position_gt.shape[0]]

    for branch in variant["order"]:
        config = variant["s2"][branch]
        sensor_indices = [ROOT_SENSOR_IDX] + list(config["local_sensor_indices"])
        imu_features = build_multi_imu_input(raw["acc"], raw["ori"], sensor_indices)
        s1_leaf_pred = s1_data[f"{branch}_p_pred"].float()

        length = min(imu_features.shape[0], s1_leaf_pred.shape[0])
        branch_input = torch.cat([imu_features[:length], s1_leaf_pred[:length]], dim=1).float().contiguous()

        branch_target = full_position_subset(full_position_gt, config["joints"])

        if branch_input.shape[1] != config["input_dim"]:
            raise RuntimeError(f"{branch}_input expected dim {config['input_dim']}, got {branch_input.shape[1]}")
        if branch_target.shape[1] != config["output_dim"]:
            raise RuntimeError(f"{branch}_target expected dim {config['output_dim']}, got {branch_target.shape[1]}")

        item[f"{branch}_input"] = branch_input
        item[f"{branch}_target"] = branch_target
        lengths.append(branch_input.shape[0])

    sequence_length = min(lengths)
    item["full_joint_position_gt"] = item["full_joint_position_gt"][:sequence_length]
    for branch in variant["order"]:
        item[f"{branch}_input"] = item[f"{branch}_input"][:sequence_length]
        item[f"{branch}_target"] = item[f"{branch}_target"][:sequence_length]
    item["num_frames"] = sequence_length

    output_path = os.path.join(output_dir, make_output_name(raw_path))
    torch.save(item, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--raw_list_file", required=True)
    parser.add_argument("--s1_pred_list_file", required=True)
    parser.add_argument("--pose_s2_gt_list_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    parser.add_argument("--skip_errors", action="store_true")
    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)

    os.makedirs(args.output_dir, exist_ok=True)

    raw_files = load_list(args.raw_list_file)
    s1_pred_files = load_list(args.s1_pred_list_file)
    gt_files = load_list(args.pose_s2_gt_list_file)

    s1_map = build_source_map(s1_pred_files, "Pose-S1 predictions")
    gt_map = build_source_map(gt_files, "Pose-S2 GT")

    saved, skipped = [], []
    for raw_path in tqdm(raw_files, desc=f"Preparing Pose-S2 data ({args.variant})", ncols=120):
        source = normalize_path(raw_path)
        if source not in s1_map or source not in gt_map:
            skipped.append((raw_path, "missing S1 pred or S2 GT"))
            continue
        try:
            saved.append(process_one(raw_path, s1_map[source], gt_map[source], variant, args.output_dir))
        except Exception as error:
            if not args.skip_errors:
                raise
            skipped.append((raw_path, repr(error)))

    output_parent = os.path.dirname(args.output_list_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(args.output_list_file, "w", encoding="utf-8") as f:
        for path in saved:
            f.write(path + "\n")

    print("\nDone.")
    print("  Raw files:", len(raw_files))
    print("  Saved:", len(saved))
    print("  Skipped:", len(skipped))
    if skipped:
        print("  Skipped examples:", skipped[:10])


if __name__ == "__main__":
    main()
