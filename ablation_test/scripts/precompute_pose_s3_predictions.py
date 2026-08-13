"""
Precompute Pose-S3 outputs for an ablation variant, assembled into
[T, 90], in the exact schema expected by the EXISTING (unmodified) fusion
trainer: scripts/train/train_pose_s3_rotation_fusion.py. That script only
reads pose3_assembled_pred_6d_reduced / pose3_target_6d_reduced /
pose_s2_full_position_input, so it is branch-count-agnostic and does not
need an ablation-specific copy.
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

from common.branch_configs import REDUCED_JOINTS, get_variant, validate_variant
from common.io_utils import load_list
from common.models import BranchRNN


import argparse
import os

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_branch_model(checkpoint_root, variant_name, branch, branch_config):
    checkpoint_path = os.path.join(checkpoint_root, branch, f"best_pose_s3_{variant_name}_{branch}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = BranchRNN(
        input_dim=int(checkpoint.get("input_dim", branch_config["input_dim"])),
        output_dim=int(checkpoint.get("output_dim", branch_config["output_dim"])),
        proj_dim=int(checkpoint.get("proj_dim", 16)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
        rnn_layers=int(checkpoint.get("rnn_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[model] {branch}: {checkpoint_path}")
    return model


@torch.inference_mode()
def infer_one_branch(model, branch_input):
    branch_input = branch_input.float().unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([branch_input.shape[1]], dtype=torch.long)
    packed = pack_padded_sequence(branch_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
    return model(packed).data.detach().cpu()


def insert_branch_prediction(assembled, branch_prediction, reduced_joints):
    branch_prediction = branch_prediction.reshape(branch_prediction.shape[0], len(reduced_joints), 6)
    assembled_view = assembled.reshape(assembled.shape[0], len(REDUCED_JOINTS), 6)
    for local_index, joint_idx in enumerate(reduced_joints):
        assembled_view[:, REDUCED_JOINTS.index(joint_idx), :] = branch_prediction[:, local_index, :]


def process_one_file(data_path, models, variant, output_dir, index):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    sequence_length = int(data["num_frames"])

    assembled = torch.zeros(sequence_length, len(REDUCED_JOINTS) * 6, dtype=torch.float32)
    branch_outputs = {}

    for branch in variant["order"]:
        branch_input = data[f"{branch}_input"].float()
        prediction = infer_one_branch(models[branch], branch_input)
        branch_outputs[branch] = prediction
        insert_branch_prediction(assembled, prediction, variant["s3"][branch]["reduced_joints"])

    output = {
        "source_file": data["source_file"],
        "pose_s2_source_file": data.get("pose_s2_source_file", ""),
        "prepared_pose_s3_source_file": data_path,
        "num_frames": sequence_length,
        "branch_order": variant["order"],
        "pose3_assembled_pred_6d_reduced": assembled,
        "pose3_target_6d_reduced": data["pose3_target_6d_reduced"].float(),
        "pose_s2_full_position_input": data["pose_s2_full_position_input"].float(),
    }

    for branch in variant["order"]:
        output[f"{branch}_pred_6d_reduced"] = branch_outputs[branch]

    output_path = os.path.join(output_dir, f"pose_s3_pred_{index:06d}.pt")
    torch.save(output, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--data_list_file", required=True, help="Output of prepare_pose_s3_data.py")
    parser.add_argument("--checkpoint_root", required=True, help="save_dir used by train_pose_s3.py for this variant")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)

    os.makedirs(args.output_dir, exist_ok=True)
    files = load_list(args.data_list_file)
    if not files:
        raise RuntimeError(f"No files found in {args.data_list_file}")

    models = {}
    print("Loading branch models:")
    for branch in variant["order"]:
        models[branch] = load_branch_model(args.checkpoint_root, args.variant, branch, variant["s3"][branch])

    saved_paths = []
    for index, data_path in enumerate(tqdm(files, desc=f"Precomputing Pose-S3 outputs ({args.variant})", ncols=120)):
        saved_paths.append(process_one_file(data_path, models, variant, args.output_dir, index))

    output_parent = os.path.dirname(args.output_list_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(args.output_list_file, "w", encoding="utf-8") as f:
        for path in saved_paths:
            f.write(path + "\n")

    print("\nDone.")
    print("  Saved:", len(saved_paths))
    print("  Output list:", args.output_list_file)

    sample = torch.load(saved_paths[0], map_location="cpu", weights_only=False)
    print("\nSample shapes:")
    print("  pose3_assembled_pred_6d_reduced:", tuple(sample["pose3_assembled_pred_6d_reduced"].shape))
    print("  pose3_target_6d_reduced:", tuple(sample["pose3_target_6d_reduced"].shape))
    print("  pose_s2_full_position_input:", tuple(sample["pose_s2_full_position_input"].shape))


if __name__ == "__main__":
    main()
