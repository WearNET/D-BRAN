"""
Precompute Pose-S1 predictions for an ablation variant, from trained
per-branch checkpoints. Mirrors scripts/precompute/precompute_pose_s1_predictions.py.

Output per sequence file: {branch}_p_pred for each branch, plus a
concatenated p_leaf_pred in variant["order"] order (matches the leaf_gt_keys
concatenation order used at training time in each branch).
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
from common.io_utils import build_multi_imu_input, load_list, make_safe_name
from common.models import BranchRNN


import argparse
import os

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_SENSOR_IDX = 5


@torch.no_grad()
def run_model(model, x):
    lengths = torch.tensor([x.shape[0]], dtype=torch.long)
    packed = pack_padded_sequence(x.unsqueeze(0).to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=True)
    out = model(packed)
    padded, _ = pad_packed_sequence(out, batch_first=True)
    return padded[0].detach().cpu()


def load_branch_model(checkpoint_root: str, variant_name: str, branch: str, branch_config):
    checkpoint_path = os.path.join(checkpoint_root, branch, f"best_pose_s1_{variant_name}_{branch}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = BranchRNN(
        input_dim=int(ckpt.get("input_dim", branch_config["input_dim"])),
        output_dim=int(ckpt.get("output_dim", branch_config["output_dim"])),
        proj_dim=int(ckpt.get("proj_dim", 32)),
        rnn_hidden=int(ckpt.get("rnn_hidden", 32)),
        rnn_layers=int(ckpt.get("rnn_layers", 2)),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] {branch}: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--raw_list_file", required=True)
    parser.add_argument("--checkpoint_root", required=True, help="save_dir used by train_pose_s1.py for this variant")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_list_file", required=True)
    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)

    os.makedirs(args.output_dir, exist_ok=True)
    files = load_list(args.raw_list_file)

    models = {
        branch: load_branch_model(args.checkpoint_root, args.variant, branch, variant["s1"][branch])
        for branch in variant["order"]
    }

    saved = []
    print("DEVICE:", DEVICE)
    print("Files:", len(files))

    for src in tqdm(files, desc=f"Pose-S1 predictions ({args.variant})", ncols=120):
        raw = torch.load(src, weights_only=False)
        data = {"source_file": src, "variant": args.variant, "branch_order": variant["order"]}

        outs = []
        for branch in variant["order"]:
            config = variant["s1"][branch]
            sensor_indices = [ROOT_SENSOR_IDX] + list(config["local_sensor_indices"])
            x = build_multi_imu_input(raw["acc"], raw["ori"], sensor_indices)
            y = run_model(models[branch], x)
            data[f"{branch}_p_pred"] = y.float()
            outs.append(y)

        data["p_leaf_pred"] = torch.cat(outs, dim=1).float()
        data["num_frames"] = data["p_leaf_pred"].shape[0]

        dst = os.path.join(args.output_dir, make_safe_name(src, "pose_s1_pred"))
        torch.save(data, dst)
        saved.append(dst)

    with open(args.output_list_file, "w") as f:
        for path in saved:
            f.write(path + "\n")

    print("Saved:", len(saved))
    print("Output list:", args.output_list_file)


if __name__ == "__main__":
    main()
