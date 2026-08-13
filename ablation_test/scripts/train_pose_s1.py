"""
Train Pose-S1 for an ablation variant (three_branch or single_branch).

Ground truth comes from the SAME files as the original five-branch D-BRAN
(data/dataset_train/pose_s1_gt_train_no_classifier/*.pt), since those files
already contain a GT leaf position per SENSOR (left_arm_p_gt, right_arm_p_gt,
left_leg_p_gt, right_leg_p_gt, head_p_gt) -- branch grouping only changes
how those per-sensor targets get combined into each branch's input/output,
not the underlying ground truth itself. No new data preparation is needed
for this stage.

Usage:
    python train_pose_s1.py --variant three_branch --branch all --save_dir <dir>
    python train_pose_s1.py --variant single_branch --branch all --save_dir <dir>
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
from main_path import POSE_S1_GT_TRAIN_LIST, POSE_S1_GT_TEST_LIST

_ablation_root = _dbran_current_file.parent.parent
if str(_ablation_root) not in _dbran_sys.path:
    _dbran_sys.path.insert(0, str(_ablation_root))

from common.branch_configs import get_variant, validate_variant
from common.io_utils import build_multi_imu_input, load_list
from common.models import BranchRNN


import argparse
import os
import random

import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PoseS1BranchDataset(Dataset):
    def __init__(self, gt_files, branch_config, root_sensor_idx: int = 5):
        self.gt_files = [p for p in gt_files if os.path.exists(p)]
        self.local_sensor_indices = branch_config["local_sensor_indices"]
        self.leaf_gt_keys = branch_config["leaf_gt_keys"]
        self.root_sensor_idx = root_sensor_idx

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, index: int):
        gt_path = self.gt_files[index]
        gt_data = torch.load(gt_path, weights_only=False)

        src_path = gt_data["source_file"]
        raw = torch.load(src_path, weights_only=False)

        sensor_indices = [self.root_sensor_idx] + list(self.local_sensor_indices)
        x = build_multi_imu_input(raw["acc"], raw["ori"], sensor_indices)

        y = torch.cat([gt_data[key].float() for key in self.leaf_gt_keys], dim=1)

        length = min(x.shape[0], y.shape[0])
        return x[:length], y[:length]


def collate_fn(batch):
    batch = sorted(batch, key=lambda z: z[0].shape[0], reverse=True)
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    return pad_sequence(xs, batch_first=True), pad_sequence(ys, batch_first=True), lengths


@torch.no_grad()
def run_eval(model, loader, criterion):
    model.eval()
    total_loss, total_batches, total_points, total_pos_error = 0.0, 0, 0, 0.0

    for x_pad, y_pad, lengths in loader:
        x_pad, y_pad = x_pad.to(DEVICE), y_pad.to(DEVICE)
        packed_x = pack_padded_sequence(x_pad, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_y = pack_padded_sequence(y_pad, lengths.cpu(), batch_first=True, enforce_sorted=True)

        pred = model(packed_x).data
        gt = packed_y.data
        loss = criterion(pred, gt)
        total_loss += loss.item()
        total_batches += 1

        num_leaves = gt.shape[1] // 3
        pos_err = torch.norm(
            (pred - gt).reshape(-1, num_leaves, 3), dim=2
        )
        total_pos_error += pos_err.sum().item()
        total_points += pos_err.numel()

    return total_loss / max(total_batches, 1), (total_pos_error / max(total_points, 1)) * 100.0


def run_train_epoch(model, loader, criterion, optimizer, scheduler, clip_val, batch_size_for_norm):
    model.train()
    running, batches = 0.0, 0

    pbar = tqdm(loader, desc="Train", ncols=120)
    for x_pad, y_pad, lengths in pbar:
        x_pad, y_pad = x_pad.to(DEVICE), y_pad.to(DEVICE)
        optimizer.zero_grad()

        packed_x = pack_padded_sequence(x_pad, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_y = pack_padded_sequence(y_pad, lengths.cpu(), batch_first=True, enforce_sorted=True)

        pred = model(packed_x).data
        gt = packed_y.data

        loss = criterion(pred, gt) / batch_size_for_norm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        scheduler.step()

        running += loss.item()
        batches += 1
        pbar.set_postfix({"Loss": f"{running / max(batches, 1):.6f}"})

    return running / max(batches, 1)


def train_branch(args, variant_name: str, branch: str, branch_config, train_gt_files, test_gt_files):
    set_seed(args.seed)

    print("\n" + "=" * 78)
    print(f"[{variant_name}] TRAINING POSE-S1 BRANCH: {branch}")
    print("=" * 78)
    print("DEVICE:", DEVICE)

    train_gt_files = list(train_gt_files)
    random.shuffle(train_gt_files)

    val_len = max(1, int(len(train_gt_files) * args.val_ratio))
    val_len = min(val_len, len(train_gt_files) - 1)
    train_files, val_files = train_gt_files[:-val_len], train_gt_files[-val_len:]

    print(f"[Data] Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_gt_files)}")

    train_ds = PoseS1BranchDataset(train_files, branch_config)
    val_ds = PoseS1BranchDataset(val_files, branch_config)
    test_ds = PoseS1BranchDataset(test_gt_files, branch_config)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin)

    input_dim = branch_config["input_dim"]
    output_dim = branch_config["output_dim"]

    model = BranchRNN(
        input_dim=input_dim, output_dim=output_dim,
        proj_dim=args.proj_dim, rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers, dropout=args.dropout,
    ).to(DEVICE)

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Input dim: {input_dim} | Output dim: {output_dim} | Parameters: {parameter_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_rate)
    criterion = torch.nn.MSELoss(reduction="sum")

    save_dir_branch = os.path.join(args.save_dir, branch)
    os.makedirs(save_dir_branch, exist_ok=True)
    best_path = os.path.join(save_dir_branch, f"best_pose_s1_{variant_name}_{branch}.pth")

    run = None
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=f"{variant_name}_{branch}",
            group=args.wandb_group or variant_name,
            config={**vars(args), "variant": variant_name, "branch": branch, "branch_config": branch_config},
            reinit=True,
        )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, scheduler, args.clip_val, args.batch_size)
        val_loss, val_pos_error_cm = run_eval(model, val_loader, criterion)

        print(f"Epoch {epoch + 1} -> TrainLoss: {train_loss:.6f} | ValLoss: {val_loss:.6f} | ValPosErr(cm): {val_pos_error_cm:.4f}")

        if run is not None:
            wandb.log({"epoch": epoch + 1, "train/loss": train_loss, "val/loss": val_loss, "val/position_error_cm": val_pos_error_cm})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "variant": variant_name,
                "branch": branch,
                "branch_config": branch_config,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "proj_dim": args.proj_dim,
                "rnn_hidden": args.rnn_hidden,
                "rnn_layers": args.rnn_layers,
                "dropout": args.dropout,
            }, best_path)
            print(f"  [*] Saved best model to: {best_path}")
        else:
            patience_counter += 1
            print(f"  [!] Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    final_val_loss, final_val_pos_error_cm = run_eval(model, val_loader, criterion)
    final_test_loss, final_test_pos_error_cm = run_eval(model, test_loader, criterion)
    print(f"\nVAL  -> Loss: {final_val_loss / args.batch_size:.6f} | PositionError(cm): {final_val_pos_error_cm:.4f}")
    print(f"TEST -> Loss: {final_test_loss / args.batch_size:.6f} | PositionError(cm): {final_test_pos_error_cm:.4f}")

    if run is not None:
        wandb.log({
            "final/val_position_error_cm": final_val_pos_error_cm,
            "final/test_position_error_cm": final_test_pos_error_cm,
        })
        wandb.finish()

    return {"val_position_error_cm": final_val_pos_error_cm, "test_position_error_cm": final_test_pos_error_cm, "parameters": parameter_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--branch", required=True, help="Branch name for the chosen variant, or 'all'")

    parser.add_argument("--train_gt_list_file", type=str, default=str(POSE_S1_GT_TRAIN_LIST))
    parser.add_argument("--test_gt_list_file", type=str, default=str(POSE_S1_GT_TEST_LIST))
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--clip_val", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--decay_rate", type=float, default=0.96)
    parser.add_argument("--decay_steps", type=int, default=2000)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    # Same recurrent width as the original five-branch Pose-S1 (32h) -- do
    # NOT scale this up for single_branch, that is the whole point of the
    # "same width" ablation.
    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--rnn_hidden", type=int, default=32)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dbran-ablation-pose-s1")
    parser.add_argument("--wandb_group", type=str, default="")

    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)
    branches = variant["order"] if args.branch == "all" else [args.branch]

    train_gt_files = load_list(args.train_gt_list_file)
    test_gt_files = load_list(args.test_gt_list_file)
    if not train_gt_files:
        raise RuntimeError("Training GT list file is empty or invalid.")
    if not test_gt_files:
        raise RuntimeError("Test GT list file is empty or invalid.")

    results = {}
    for branch in branches:
        results[branch] = train_branch(args, args.variant, branch, variant["s1"][branch], train_gt_files, test_gt_files)

    print("\n" + "=" * 78)
    print(f"FINAL POSE-S1 RESULTS ({args.variant})")
    print("=" * 78)
    for branch, metrics in results.items():
        print(f"{branch:>12}: ValPos={metrics['val_position_error_cm']:.4f} cm | TestPos={metrics['test_position_error_cm']:.4f} cm | Params={metrics['parameters']}")


if __name__ == "__main__":
    main()
