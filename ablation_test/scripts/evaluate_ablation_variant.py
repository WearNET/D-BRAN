"""
Full-pipeline offline evaluator for an ablation variant (three_branch or
single_branch), producing the Table IV row: SIP / Ang / Mesh / Jitter /
Par.(M). Mirrors scripts/profile/profile_full_pipeline_fivebranch.py's
evaluation logic (same PoseEvaluator formula, same reduced-pose ->
full-pose reconstruction), generalized via common/branch_configs.py
instead of the five-branch structure hardcoded there.

Runs S1 -> S2 -> S3 -> fusion -> full pose for every sequence, end to end,
from the trained checkpoints produced by this folder's train_pose_s1/s2/s3.py
and the ORIGINAL (unmodified) scripts/train/train_pose_s3_rotation_fusion.py.

Usage:
    python evaluate_ablation_variant.py --variant three_branch \
        --checkpoint_root checkpoints/retrained/ablation_three_branch \
        --raw_list_file data/dataset_train/test.txt
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
from config import paths, joint_set
from net import TransPoseNet
import articulate as art

_ablation_root = _dbran_current_file.parent.parent
if str(_ablation_root) not in _dbran_sys.path:
    _dbran_sys.path.insert(0, str(_ablation_root))

from common.branch_configs import REDUCED_JOINTS, get_variant, validate_variant
from common.io_utils import build_multi_imu_input, full_position_subset, load_list
from common.models import BranchRNN, FusionRNN


import argparse
import json
import os

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_SENSOR_IDX = 5


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------
def load_stage_branch_model(checkpoint_root, stage, variant_name, branch, branch_config,
                             default_proj_dim, default_rnn_hidden, default_rnn_layers):
    checkpoint_path = os.path.join(checkpoint_root, stage, branch, f"best_{stage}_{variant_name}_{branch}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Train {stage} for "
            f"variant='{variant_name}' branch='{branch}' first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = BranchRNN(
        input_dim=int(checkpoint.get("input_dim", branch_config["input_dim"])),
        output_dim=int(checkpoint.get("output_dim", branch_config["output_dim"])),
        proj_dim=int(checkpoint.get("proj_dim", default_proj_dim)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", default_rnn_hidden)),
        rnn_layers=int(checkpoint.get("rnn_layers", default_rnn_layers)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint_path


def load_fusion_model(checkpoint_root):
    checkpoint_path = os.path.join(checkpoint_root, "fusion", "best_pose_s3_five_branch_rotation_fusion.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No fusion checkpoint found at {checkpoint_path}. Train fusion first.")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    use_pose_s2_position = bool(checkpoint.get("use_pose_s2_position", False))
    input_dim = int(checkpoint.get("input_dim", 90 + 69 if use_pose_s2_position else 90))

    model = FusionRNN(
        input_dim=input_dim,
        output_dim=int(checkpoint.get("output_dim", 90)),
        proj_dim=int(checkpoint.get("proj_dim", 16)),
        rnn_hidden=int(checkpoint.get("rnn_hidden", 16)),
        rnn_layers=int(checkpoint.get("rnn_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, use_pose_s2_position, checkpoint_path


def load_all_models(checkpoint_root, variant_name, variant):
    s1_models, s2_models, s3_models = {}, {}, {}

    print("Loading checkpoints:")
    for branch in variant["order"]:
        model, path = load_stage_branch_model(checkpoint_root, "pose_s1", variant_name, branch, variant["s1"][branch], 32, 32, 2)
        s1_models[branch] = model
        print(f"  [pose_s1] {branch}: {path}")

    for branch in variant["order"]:
        model, path = load_stage_branch_model(checkpoint_root, "pose_s2", variant_name, branch, variant["s2"][branch], 16, 16, 2)
        s2_models[branch] = model
        print(f"  [pose_s2] {branch}: {path}")

    for branch in variant["order"]:
        model, path = load_stage_branch_model(checkpoint_root, "pose_s3", variant_name, branch, variant["s3"][branch], 16, 16, 2)
        s3_models[branch] = model
        print(f"  [pose_s3] {branch}: {path}")

    fusion_model, use_pose_s2_position, fusion_path = load_fusion_model(checkpoint_root)
    print(f"  [fusion]  {fusion_path} (use_pose_s2_position={use_pose_s2_position})")

    return s1_models, s2_models, s3_models, fusion_model, use_pose_s2_position


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
@torch.no_grad()
def run_branch_inference(model, x: torch.Tensor) -> torch.Tensor:
    lengths = torch.tensor([x.shape[0]], dtype=torch.long)
    packed = pack_padded_sequence(x.unsqueeze(0).to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=True)
    out = model(packed)
    padded, _ = pad_packed_sequence(out, batch_first=True)
    return padded[0].detach().cpu()


def insert_position_prediction(assembled, branch_prediction, joints):
    prediction = branch_prediction.reshape(branch_prediction.shape[0], len(joints), 3)
    assembled_view = assembled.reshape(assembled.shape[0], 23, 3)
    for local_index, joint_idx in enumerate(joints):
        assembled_view[:, joint_idx - 1, :] = prediction[:, local_index, :]


def insert_reduced_prediction(assembled, branch_prediction, reduced_joints):
    prediction = branch_prediction.reshape(branch_prediction.shape[0], len(reduced_joints), 6)
    assembled_view = assembled.reshape(assembled.shape[0], len(REDUCED_JOINTS), 6)
    for local_index, joint_idx in enumerate(reduced_joints):
        assembled_view[:, REDUCED_JOINTS.index(joint_idx), :] = prediction[:, local_index, :]


@torch.inference_mode()
def distributed_pose_forward(raw, variant, s1_models, s2_models, s3_models, fusion_model, use_pose_s2_position, trans_net):
    acc, ori = raw["acc"], raw["ori"]

    # Pose-S1
    s1_pred = {}
    for branch in variant["order"]:
        config = variant["s1"][branch]
        sensor_indices = [ROOT_SENSOR_IDX] + list(config["local_sensor_indices"])
        x = build_multi_imu_input(acc, ori, sensor_indices)
        s1_pred[branch] = run_branch_inference(s1_models[branch], x)

    length = min(t.shape[0] for t in s1_pred.values())
    s1_pred = {k: v[:length] for k, v in s1_pred.items()}

    # Pose-S2
    p_full = torch.zeros(length, 69, dtype=torch.float32)
    for branch in variant["order"]:
        config = variant["s2"][branch]
        sensor_indices = [ROOT_SENSOR_IDX] + list(config["local_sensor_indices"])
        imu = build_multi_imu_input(acc, ori, sensor_indices)[:length]
        branch_input = torch.cat([imu, s1_pred[branch][:length]], dim=1)
        branch_pred = run_branch_inference(s2_models[branch], branch_input)
        insert_position_prediction(p_full, branch_pred[:length], config["joints"])

    # Pose-S3
    assembled_reduced = torch.zeros(length, len(REDUCED_JOINTS) * 6, dtype=torch.float32)
    for branch in variant["order"]:
        config = variant["s3"][branch]
        imu = build_multi_imu_input(acc, ori, config["sensor_indices"])[:length]
        position_features = full_position_subset(p_full, config["position_joints"])[:length]
        branch_input = torch.cat([imu, position_features], dim=1)
        branch_pred = run_branch_inference(s3_models[branch], branch_input)
        insert_reduced_prediction(assembled_reduced, branch_pred[:length], config["reduced_joints"])

    # Fusion
    if use_pose_s2_position:
        fusion_input = torch.cat([assembled_reduced, p_full], dim=1)
    else:
        fusion_input = assembled_reduced
    delta = run_branch_inference(fusion_model, fusion_input)
    fusion_length = min(assembled_reduced.shape[0], delta.shape[0])
    reduced_pose = assembled_reduced[:fusion_length] + delta[:fusion_length]

    # Reduced 6D -> full 24-joint local rotation matrices (purely geometric,
    # no learned weights involved -- see net.py's global_to_local_pose).
    root_rotation = ori[:fusion_length, ROOT_SENSOR_IDX].reshape(fusion_length, 3, 3)
    pose = trans_net._reduced_glb_6d_to_full_local_mat(root_rotation.cpu(), reduced_pose.cpu())

    return pose


# ------------------------------------------------------------
# Evaluation (same formula as scripts/profile/profile_full_pipeline_fivebranch.py)
# ------------------------------------------------------------
class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        length = min(pose_p.shape[0], pose_t.shape[0])
        pose_p = pose_p[:length].clone().view(-1, 24, 3, 3)
        pose_t = pose_t[:length].clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors, title):
        print(f"\n========== {title} ==========")
        names = ["SIP Error (deg)", "Angular Error (deg)", "Positional Error (cm)", "Mesh Error (cm)", "Jitter Error (100m/s^3)"]
        for index, name in enumerate(names):
            print("%s: %.2f (+/- %.2f)" % (name, float(errors[index, 0]), float(errors[index, 1])))


def summarize(errs, title):
    # Each entry in errs is already [5, 2] (mean, std) per metric, computed
    # WITHIN that sequence by art.FullMotionEvaluator. Averaging those
    # pairs across sequences (not recomputing a new std) matches Table
    # II/III/IV's convention in profile_full_pipeline_fivebranch.py.
    all_errs = torch.stack(errs, dim=0)
    errors = all_errs.mean(dim=0)
    PoseEvaluator.print(errors, title)
    return errors


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["three_branch", "single_branch"])
    parser.add_argument("--checkpoint_root", required=True, help="e.g. checkpoints/retrained/ablation_three_branch")
    parser.add_argument("--raw_list_file", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--export_per_sequence", type=str, default=None)
    args = parser.parse_args()

    raw_list_file = args.raw_list_file or str(PROJECT_ROOT / "data" / "dataset_train" / "test.txt")

    validate_variant(args.variant)
    variant = get_variant(args.variant)

    print("DEVICE:", DEVICE)
    print("Variant:", args.variant)
    print("Branch order:", variant["order"])

    s1_models, s2_models, s3_models, fusion_model, use_pose_s2_position = load_all_models(args.checkpoint_root, args.variant, variant)

    s1_params = sum(count_params(m) for m in s1_models.values())
    s2_params = sum(count_params(m) for m in s2_models.values())
    s3_params = sum(count_params(m) for m in s3_models.values())
    fusion_params = count_params(fusion_model)
    total_params = s1_params + s2_params + s3_params + fusion_params

    print("\n==================== SIZE STATS ====================")
    print(f"  Pose-S1 params:  {s1_params}")
    print(f"  Pose-S2 params:  {s2_params}")
    print(f"  Pose-S3 params:  {s3_params}")
    print(f"  Fusion params:   {fusion_params}")
    print(f"  Total (M):       {total_params / 1e6:.4f}")

    trans_net = TransPoseNet(num_past_frame=20, num_future_frame=5).to(DEVICE)
    trans_net.eval()

    raw_paths = load_list(raw_list_file)
    if args.max_sequences is not None:
        raw_paths = raw_paths[:args.max_sequences]
    if not raw_paths:
        raise RuntimeError(f"No sequences found in {raw_list_file}")

    evaluator = PoseEvaluator()
    errs = []
    per_sequence_records = []

    for raw_path in tqdm(raw_paths, desc=f"Evaluating {args.variant}", ncols=120):
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        pose_t = art.math.axis_angle_to_rotation_matrix(raw["pose"]).view(-1, 24, 3, 3)

        pose_p = distributed_pose_forward(raw, variant, s1_models, s2_models, s3_models, fusion_model, use_pose_s2_position, trans_net)

        error_vec = evaluator.eval(pose_p, pose_t)
        errs.append(error_vec)

        if args.export_per_sequence is not None:
            per_sequence_records.append({
                "path": raw_path,
                "sip": float(error_vec[0, 0]),
                "ang": float(error_vec[1, 0]),
                "pos": float(error_vec[2, 0]),
                "mesh": float(error_vec[3, 0]),
                "jitter": float(error_vec[4, 0]),
            })

    print("\n==================== OFFLINE EVALUATION ====================")
    errors = summarize(errs, f"Ablation variant: {args.variant}")

    print("\n==================== TABLE IV ROW ====================")
    print(f"{args.variant:<20} SIP={float(errors[0,0]):.2f} | Ang={float(errors[1,0]):.2f} | "
          f"Mesh={float(errors[3,0]):.2f} | Jitter={float(errors[4,0]):.2f} | "
          f"Par.(M)={total_params / 1e6:.2f}")

    if args.export_per_sequence is not None:
        export_path = _DbranPath(args.export_per_sequence)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(per_sequence_records, f, indent=2)
        print(f"\n[info] Wrote per-sequence errors for {len(per_sequence_records)} sequences to {export_path}")


if __name__ == "__main__":
    main()
