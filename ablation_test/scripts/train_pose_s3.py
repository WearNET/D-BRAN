"""
Train Pose-S3 for an ablation variant, from files produced by
prepare_pose_s3_data.py. Mirrors scripts/train/train_pose_s3_region.py.
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
from common.io_utils import angular_error_deg, load_list
from common.models import BranchRNN


import argparse
import os
import random

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseS3BranchDataset(Dataset):
    def __init__(self, files, branch: str, imu_dim: int, training: bool, position_noise_std: float):
        self.files = list(files)
        self.branch = branch
        self.imu_dim = int(imu_dim)
        self.training = training
        self.position_noise_std = float(position_noise_std)
        self.input_key = f"{branch}_input"
        self.target_key = f"{branch}_target_6d_reduced"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        data = torch.load(self.files[index], map_location="cpu", weights_only=False)
        branch_input = data[self.input_key].float().clone()
        branch_target = data[self.target_key].float()

        if self.training and self.position_noise_std > 0:
            branch_input[:, self.imu_dim:] += torch.randn_like(branch_input[:, self.imu_dim:]) * self.position_noise_std

        length = min(branch_input.shape[0], branch_target.shape[0])
        return branch_input[:length], branch_target[:length], length


def collate_batch(batch):
    inputs, targets, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return pad_sequence(inputs, batch_first=True), pad_sequence(targets, batch_first=True), lengths


def run_epoch(model, loader, criterion, optimizer, grad_clip, num_joints, training):
    model.train(training)
    total_loss, total_angular, total_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for inputs, targets, lengths in tqdm(loader, desc="Train" if training else "Eval", ncols=120):
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            packed_inputs = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True, enforce_sorted=False)

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(packed_inputs)
            loss = criterion(prediction.data, packed_targets.data)
            angular = angular_error_deg(prediction.data, packed_targets.data, num_joints)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.item())
            total_angular += float(angular.item())
            total_batches += 1

    return {"loss": total_loss / max(total_batches, 1), "angular_error_deg": total_angular / max(total_batches, 1)}


def train_one_branch(args, variant_name, branch, branch_config, train_files, test_files):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dim, output_dim = branch_config["input_dim"], branch_config["output_dim"]
    num_joints = output_dim // 6
    imu_dim = 12 * len(branch_config["sensor_indices"])

    full_train = PoseS3BranchDataset(train_files, branch, imu_dim, training=True, position_noise_std=args.position_noise_std)
    test_dataset = PoseS3BranchDataset(test_files, branch, imu_dim, training=False, position_noise_std=0.0)

    n_total = len(full_train)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_train, [n_train, n_val], generator=generator)
    val_dataset.dataset.training = False

    loader_kwargs = {"batch_size": args.batch_size, "collate_fn": collate_batch, "num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = BranchRNN(input_dim=input_dim, output_dim=output_dim, proj_dim=args.proj_dim, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, dropout=args.dropout).to(DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters())

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_branch_dir = os.path.join(args.save_dir, branch)
    os.makedirs(save_branch_dir, exist_ok=True)
    best_path = os.path.join(save_branch_dir, f"best_pose_s3_{variant_name}_{branch}.pth")

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=f"{variant_name}_{branch}_{args.rnn_hidden}h", group=args.wandb_group or variant_name, config={**vars(args), "variant": variant_name, "branch": branch, "branch_config": branch_config}, reinit=True)

    print("\n" + "=" * 80)
    print(f"[{variant_name}] Training Pose-S3 branch: {branch}")
    print("=" * 80)
    print(f"Device: {DEVICE} | Input dim: {input_dim} | Output dim: {output_dim} | Reduced joints: {num_joints} | Parameters: {parameter_count}")
    print(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    best_val_angular = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, args.grad_clip, num_joints, training=True)
        val_metrics = run_epoch(model, val_loader, criterion, None, args.grad_clip, num_joints, training=False)

        print(f"Epoch {epoch:03d} -> TrainLoss={train_metrics['loss']:.6f} | TrainAng={train_metrics['angular_error_deg']:.4f} deg | ValLoss={val_metrics['loss']:.6f} | ValAng={val_metrics['angular_error_deg']:.4f} deg")

        if run is not None:
            import wandb
            wandb.log({"epoch": epoch, "train/loss": train_metrics["loss"], "train/angular_error_deg": train_metrics["angular_error_deg"], "val/loss": val_metrics["loss"], "val/angular_error_deg": val_metrics["angular_error_deg"]})

        if val_metrics["angular_error_deg"] < best_val_angular:
            best_val_angular = val_metrics["angular_error_deg"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(), "variant": variant_name, "branch": branch,
                "branch_config": branch_config, "input_dim": input_dim, "output_dim": output_dim,
                "proj_dim": args.proj_dim, "rnn_hidden": args.rnn_hidden, "rnn_layers": args.rnn_layers,
                "dropout": args.dropout, "parameter_count": parameter_count,
                "best_val_angular_error_deg": best_val_angular, "epoch": epoch,
            }, best_path)
            print(f"  [*] Saved best model to: {best_path}")
        else:
            patience_counter += 1
            print(f"  [!] Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = run_epoch(model, val_loader, criterion, None, args.grad_clip, num_joints, training=False)
    test_metrics = run_epoch(model, test_loader, criterion, None, args.grad_clip, num_joints, training=False)
    print(f"\nVAL  -> Loss={val_metrics['loss']:.6f} | Angular={val_metrics['angular_error_deg']:.4f} deg")
    print(f"TEST -> Loss={test_metrics['loss']:.6f} | Angular={test_metrics['angular_error_deg']:.4f} deg")

    if run is not None:
        import wandb
        wandb.log({"final/val_angular_error_deg": val_metrics["angular_error_deg"], "final/test_angular_error_deg": test_metrics["angular_error_deg"]})
        wandb.finish()

    return {"val_angular_error_deg": val_metrics["angular_error_deg"], "test_angular_error_deg": test_metrics["angular_error_deg"], "parameters": parameter_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--branch", required=True)
    parser.add_argument("--train_list_file", required=True)
    parser.add_argument("--test_list_file", required=True)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden", type=int, default=16)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--position_noise_std", type=float, default=0.025)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="dbran-ablation-pose-s3")
    parser.add_argument("--wandb_group", default=None)

    args = parser.parse_args()

    validate_variant(args.variant)
    variant = get_variant(args.variant)
    branches = variant["order"] if args.branch == "all" else [args.branch]

    train_files = load_list(args.train_list_file)
    test_files = load_list(args.test_list_file)
    if not train_files:
        raise RuntimeError(f"No training files found in {args.train_list_file}")
    if not test_files:
        raise RuntimeError(f"No test files found in {args.test_list_file}")

    results = {}
    for branch in branches:
        results[branch] = train_one_branch(args, args.variant, branch, variant["s3"][branch], train_files, test_files)

    print("\n" + "=" * 80)
    print(f"FINAL POSE-S3 RESULTS ({args.variant})")
    print("=" * 80)
    for branch, metrics in results.items():
        print(f"{branch:>12}: ValAng={metrics['val_angular_error_deg']:.4f} deg | TestAng={metrics['test_angular_error_deg']:.4f} deg | Params={metrics['parameters']}")


if __name__ == "__main__":
    main()
