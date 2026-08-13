"""
Train Pose-S2 for an ablation variant, from files produced by
prepare_pose_s2_data.py.

Usage:
    python train_pose_s2.py --variant three_branch --branch all \
        --train_list_file <...> --test_list_file <...> --save_dir <dir>
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
import random

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
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


class PoseS2BranchDataset(Dataset):
    def __init__(self, files, branch: str, imu_dim: int, training: bool, leaf_noise_sigma: float):
        self.files = list(files)
        self.branch = branch
        self.imu_dim = int(imu_dim)
        self.training = training
        self.leaf_noise_sigma = float(leaf_noise_sigma)
        self.input_key = f"{branch}_input"
        self.target_key = f"{branch}_target"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        data = torch.load(self.files[index], map_location="cpu", weights_only=False)
        x = data[self.input_key].float().clone()
        y = data[self.target_key].float()

        if self.training and self.leaf_noise_sigma > 0:
            # Leaf-prediction features are the tail of the input (after the
            # 12*N IMU features) -- mirrors the five-branch design's noise
            # augmentation on the S1 leaf input.
            x[:, self.imu_dim:] += torch.randn_like(x[:, self.imu_dim:]) * self.leaf_noise_sigma

        length = min(x.shape[0], y.shape[0])
        return x[:length], y[:length], length


def collate_fn(batch):
    xs, ys, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return pad_sequence(xs, batch_first=True), pad_sequence(ys, batch_first=True), lengths


def run_epoch(model, loader, criterion, optimizer, clip_val, training):
    model.train(training)
    total_loss, total_pos_error, total_points, total_batches = 0.0, 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for x_pad, y_pad, lengths in tqdm(loader, desc="Train" if training else "Eval", ncols=120):
            x_pad, y_pad = x_pad.to(DEVICE), y_pad.to(DEVICE)
            packed_x = pack_padded_sequence(x_pad, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_y = pack_padded_sequence(y_pad, lengths.cpu(), batch_first=True, enforce_sorted=False)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred = model(packed_x).data
            gt = packed_y.data
            loss = criterion(pred, gt)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()

            num_joints = gt.shape[1] // 3
            pos_err = torch.norm((pred - gt).reshape(-1, num_joints, 3), dim=2)
            total_pos_error += pos_err.sum().item()
            total_points += pos_err.numel()
            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1), (total_pos_error / max(total_points, 1)) * 100.0


def train_branch(args, variant_name, branch, branch_config, train_files_all, test_files):
    set_seed(args.seed)

    shuffled = list(train_files_all)
    random.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * args.val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    train_files, val_files = shuffled[:-val_count], shuffled[-val_count:]

    print("\n" + "=" * 78)
    print(f"[{variant_name}] TRAINING POSE-S2 BRANCH: {branch}")
    print("=" * 78)
    print(f"[Data] Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    imu_dim = 12 * (1 + len(branch_config["local_sensor_indices"]))

    train_dataset = PoseS2BranchDataset(train_files, branch, imu_dim, training=True, leaf_noise_sigma=args.leaf_noise_sigma)
    val_dataset = PoseS2BranchDataset(val_files, branch, imu_dim, training=False, leaf_noise_sigma=0.0)
    test_dataset = PoseS2BranchDataset(test_files, branch, imu_dim, training=False, leaf_noise_sigma=0.0)

    loader_kwargs = {"batch_size": args.batch_size, "collate_fn": collate_fn, "num_workers": args.num_workers, "pin_memory": DEVICE.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    input_dim, output_dim = branch_config["input_dim"], branch_config["output_dim"]

    model = BranchRNN(input_dim=input_dim, output_dim=output_dim, proj_dim=args.proj_dim, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, dropout=args.dropout).to(DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Input dim: {input_dim} | Output dim: {output_dim} | Joints: {branch_config['joints']} | Parameters: {parameter_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_rate)
    criterion = torch.nn.MSELoss()

    save_dir_branch = os.path.join(args.save_dir, branch)
    os.makedirs(save_dir_branch, exist_ok=True)
    checkpoint_path = os.path.join(save_dir_branch, f"best_pose_s2_{variant_name}_{branch}.pth")

    run = None
    if args.use_wandb:
        run = wandb.init(project=args.wandb_project, name=f"{variant_name}_{branch}", group=args.wandb_group or variant_name, config={**vars(args), "variant": variant_name, "branch": branch, "branch_config": branch_config}, reinit=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_pos = run_epoch(model, train_loader, criterion, optimizer, args.clip_val, training=True)
        val_loss, val_pos = run_epoch(model, val_loader, criterion, None, args.clip_val, training=False)
        scheduler.step()

        print(f"Epoch {epoch:03d} -> TrainLoss: {train_loss:.6f} | TrainPos(cm): {train_pos:.4f} | ValLoss: {val_loss:.6f} | ValPos(cm): {val_pos:.4f}")

        if run is not None:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "train/position_error_cm": train_pos, "val/loss": val_loss, "val/position_error_cm": val_pos})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(), "variant": variant_name, "branch": branch,
                "branch_config": branch_config, "input_dim": input_dim, "output_dim": output_dim,
                "proj_dim": args.proj_dim, "rnn_hidden": args.rnn_hidden, "rnn_layers": args.rnn_layers,
                "dropout": args.dropout, "parameter_count": parameter_count,
                "best_validation_loss": best_val_loss, "epoch": epoch,
            }, checkpoint_path)
            print(f"  [*] Saved best model to: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  [!] Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_val_loss, final_val_pos = run_epoch(model, val_loader, criterion, None, args.clip_val, training=False)
    final_test_loss, final_test_pos = run_epoch(model, test_loader, criterion, None, args.clip_val, training=False)
    print(f"\nVAL  -> Loss: {final_val_loss:.6f} | PositionError(cm): {final_val_pos:.4f}")
    print(f"TEST -> Loss: {final_test_loss:.6f} | PositionError(cm): {final_test_pos:.4f}")

    if run is not None:
        wandb.log({"final/val_position_error_cm": final_val_pos, "final/test_position_error_cm": final_test_pos})
        wandb.finish()

    return {"val_position_error_cm": final_val_pos, "test_position_error_cm": final_test_pos, "parameters": parameter_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--branch", required=True, help="Branch name for the chosen variant, or 'all'")
    parser.add_argument("--train_list_file", required=True)
    parser.add_argument("--test_list_file", required=True)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden", type=int, default=16)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--clip_val", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--decay_rate", type=float, default=0.96)
    parser.add_argument("--decay_steps", type=int, default=2000)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--leaf_noise_sigma", type=float, default=0.04)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="dbran-ablation-pose-s2")
    parser.add_argument("--wandb_group", default="")

    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)
    branches = variant["order"] if args.branch == "all" else [args.branch]

    train_files = load_list(args.train_list_file)
    test_files = load_list(args.test_list_file)
    if len(train_files) < 2:
        raise RuntimeError("The training list must contain at least two valid files.")
    if not test_files:
        raise RuntimeError("The test list contains no valid files.")

    results = {}
    for branch in branches:
        results[branch] = train_branch(args, args.variant, branch, variant["s2"][branch], train_files, test_files)

    print("\n" + "=" * 78)
    print(f"FINAL POSE-S2 RESULTS ({args.variant})")
    print("=" * 78)
    for branch, metrics in results.items():
        print(f"{branch:>12}: ValPos={metrics['val_position_error_cm']:.4f} cm | TestPos={metrics['test_position_error_cm']:.4f} cm | Params={metrics['parameters']}")


if __name__ == "__main__":
    main()
