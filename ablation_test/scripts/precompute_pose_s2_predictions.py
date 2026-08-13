"""
Precompute Pose-S2 outputs for an ablation variant, assembled into the
standard [T, 69] full joint-position vector. Mirrors
scripts/precompute/precompute_pose_s2_full_distributed_outputs.py.
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
from common.io_utils import load_list
from common.models import BranchRNN


import argparse
import os
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_branch_model(checkpoint_root, variant_name, branch, branch_config):
    checkpoint_path = os.path.join(checkpoint_root, branch, f"best_pose_s2_{variant_name}_{branch}.pth")
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
def infer_branch(model, branch_input):
    branch_input = branch_input.float().unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([branch_input.shape[1]], dtype=torch.long)
    packed = pack_padded_sequence(branch_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
    return model(packed).data.detach().cpu()


def insert_branch_prediction(assembled, branch_prediction, joints):
    if branch_prediction.shape[1] != len(joints) * 3:
        raise ValueError(f"Prediction dim {branch_prediction.shape[1]} does not match {len(joints)} joints.")
    prediction = branch_prediction.reshape(branch_prediction.shape[0], len(joints), 3)
    assembled_view = assembled.reshape(assembled.shape[0], 23, 3)
    for local_index, joint_idx in enumerate(joints):
        assembled_view[:, joint_idx - 1, :] = prediction[:, local_index, :]


def process_one_file(prepared_path, models, variant, output_dir, index):
    data = torch.load(prepared_path, map_location="cpu", weights_only=False)

    sequence_lengths = [data[f"{branch}_input"].shape[0] for branch in variant["order"]]
    sequence_length = min(sequence_lengths)

    assembled = torch.zeros(sequence_length, 69, dtype=torch.float32)
    branch_predictions = {}

    for branch in variant["order"]:
        branch_input = data[f"{branch}_input"][:sequence_length].float()
        prediction = infer_branch(models[branch], branch_input)
        branch_predictions[branch] = prediction
        insert_branch_prediction(assembled, prediction, variant["s2"][branch]["joints"])

    output = {
        "source_file": data["source_file"],
        "prepared_pose_s2_source_file": prepared_path,
        "num_frames": sequence_length,
        "branch_order": variant["order"],
        "full_joint_position_pred": assembled,
    }
    if "full_joint_position_gt" in data:
        output["full_joint_position_gt"] = data["full_joint_position_gt"][:sequence_length].float()

    for branch in variant["order"]:
        output[f"{branch}_pred"] = branch_predictions[branch]

    output_path = os.path.join(output_dir, f"pose_s2_pred_{index:06d}.pt")
    torch.save(output, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--data_list_file", required=True, help="Output of prepare_pose_s2_data.py")
    parser.add_argument("--checkpoint_root", required=True, help="save_dir used by train_pose_s2.py for this variant")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)

    os.makedirs(args.output_dir, exist_ok=True)
    prepared_files = load_list(args.data_list_file)
    if not prepared_files:
        raise RuntimeError(f"No prepared files found in {args.data_list_file}")

    models = {branch: load_branch_model(args.checkpoint_root, args.variant, branch, variant["s2"][branch]) for branch in variant["order"]}

    saved_paths = []
    for index, prepared_path in enumerate(tqdm(prepared_files, desc=f"Precomputing Pose-S2 outputs ({args.variant})", ncols=120)):
        saved_paths.append(process_one_file(prepared_path, models, variant, args.output_dir, index))

    output_parent = os.path.dirname(args.output_list_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(args.output_list_file, "w", encoding="utf-8") as f:
        for path in saved_paths:
            f.write(path + "\n")

    print("\nDone.")
    print("  Saved:", len(saved_paths))
    print("  Output list:", args.output_list_file)


if __name__ == "__main__":
    main()
