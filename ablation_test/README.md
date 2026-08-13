# Ablation test infrastructure (temporary, not for the main repo)

This folder is self-contained on purpose: everything needed to train and
evaluate the paper's two remaining Table IV ablation variants lives here,
separate from `scripts/`, so it is easy to identify and drop before the
`D-BRAN` repo gets pushed anywhere public. Nothing under `scripts/`,
`checkpoints/`, or `data/` was modified to build this.

## What this produces

Table IV needs three rows beyond "D-BRAN (full)" and "w/o residual fusion"
(the latter was already obtained via a profiler flag, no retraining
needed):

- **Three-branch partition** (legs, trunk-head, arms)
- **Single branch, same width**

Both variants train on the **exact same protocol as the original D-BRAN**
(AMASS + DIP-IMU, no custom OptiTrack/Xsens data), so Table IV stays a
clean architecture-only comparison against the existing pristine
checkpoints. The custom-data experiment agreed separately (D-BRAN five-branch
+ your OptiTrack/Xsens captures) is unrelated to this folder and uses the
augmented manifests already sitting in `data/dataset_train/*_augmented.txt`.

## Why so few new files

Digging through `scripts/data` and `scripts/train` first paid off: the
existing Pose-S1 and Pose-S2 ground-truth files
(`pose_s1_gt_train_no_classifier/*.pt`, `pose_s2_gt_train/*.pt`) already
store **per-sensor / per-joint** targets, not sliced to the five-branch
grouping. That data is reused as-is — no new ground-truth preparation was
needed for S1 or S2. The learned residual fusion network is also reused
completely unchanged (`scripts/train/train_pose_s3_rotation_fusion.py`):
its input/output is always the same 90-dim vector regardless of how many
branches produced it.

What genuinely needed new code: the per-branch **input construction**
(which sensors feed which network) and **output slicing** (which joints
each network predicts), since those depend on the branch grouping. That
lives in `common/branch_configs.py`.

## Structure

```
ablation_test/
├── common/
│   ├── branch_configs.py   # sensor/joint groupings for both variants + a
│   │                         self-check (`python branch_configs.py`)
│   ├── io_utils.py          # shared tensor-building helpers
│   └── models.py             # generic BranchRNN / FusionRNN (same structure
│                               as the originals, just parameterized dims)
└── scripts/
    ├── train_pose_s1.py
    ├── precompute_pose_s1_predictions.py
    ├── prepare_pose_s2_data.py
    ├── train_pose_s2.py
    ├── precompute_pose_s2_predictions.py
    ├── prepare_pose_s3_data.py
    ├── train_pose_s3.py
    ├── precompute_pose_s3_predictions.py
    └── evaluate_ablation_variant.py   # Table IV row: SIP/Ang/Mesh/Jitter/Par.(M)
```

Every script takes `--variant three_branch|single_branch` and (where
applicable) `--branch <name>|all`. Branch names: `legs`, `trunk_head`,
`arms` for three_branch; `single` for single_branch.

Recurrent widths default to the same values as the original five-branch
checkpoints (Pose-S1: proj_dim=rnn_hidden=32; Pose-S2/S3: 16) — this is
intentional, not a placeholder. Do not raise them for `single_branch`; the
whole point of that row is to isolate the anatomical-partition effect from
network capacity.

Validated end-to-end (both variants, all 4 stages, including the
unmodified fusion trainer) on a 2-sequence smoke test before handing this
off — see the parameter counts below to confirm a real run picked up the
right config: fusion always reports **8778 parameters**, identical to the
original five-branch fusion, since it is the exact same network.

## Exact command sequence (per variant)

Replace `VARIANT` with `three_branch` or `single_branch`, and
`CKPT=checkpoints/retrained/ablation_$VARIANT` (keep this OUTSIDE
`checkpoints/dbran_*` so it never collides with the pristine checkpoints
that reproduce the paper's Table II/III numbers).

```bash
VARIANT=three_branch   # or single_branch
CKPT=checkpoints/retrained/ablation_$VARIANT

# 1) Pose-S1 -- ground truth is the EXISTING manifests, unchanged.
python ablation_test/scripts/train_pose_s1.py \
  --variant $VARIANT --branch all \
  --save_dir $CKPT/pose_s1

# 2) Precompute Pose-S1 predictions on train+test raw sequences.
python ablation_test/scripts/precompute_pose_s1_predictions.py \
  --variant $VARIANT \
  --raw_list_file data/dataset_train/train_pose.txt \
  --checkpoint_root $CKPT/pose_s1 \
  --output_dir data/dataset_train/ablation_${VARIANT}_pose_s1_pred_train \
  --output_list_file data/dataset_train/ablation_${VARIANT}_pose_s1_pred_train.txt
# repeat with --raw_list_file data/dataset_train/test.txt (swap train->test above)

# 3) Prepare Pose-S2 data (S1 pred + existing S2 GT -> per-branch tensors).
python ablation_test/scripts/prepare_pose_s2_data.py \
  --variant $VARIANT \
  --raw_list_file data/dataset_train/train_pose.txt \
  --s1_pred_list_file data/dataset_train/ablation_${VARIANT}_pose_s1_pred_train.txt \
  --pose_s2_gt_list_file data/dataset_train/train_pose_s2_gt.txt \
  --output_dir data/dataset_train/ablation_${VARIANT}_pose_s2_data_train \
  --output_list_file data/dataset_train/ablation_${VARIANT}_pose_s2_data_train.txt
# repeat for test (test.txt / pose_s1_pred_test / test_pose_s2_gt.txt)

# 4) Pose-S2
python ablation_test/scripts/train_pose_s2.py \
  --variant $VARIANT --branch all \
  --train_list_file data/dataset_train/ablation_${VARIANT}_pose_s2_data_train.txt \
  --test_list_file data/dataset_train/ablation_${VARIANT}_pose_s2_data_test.txt \
  --save_dir $CKPT/pose_s2

# 5) Precompute Pose-S2 outputs (assembled [T,69])
python ablation_test/scripts/precompute_pose_s2_predictions.py \
  --variant $VARIANT \
  --data_list_file data/dataset_train/ablation_${VARIANT}_pose_s2_data_train.txt \
  --checkpoint_root $CKPT/pose_s2 \
  --output_dir data/dataset_train/ablation_${VARIANT}_pose_s2_pred_train \
  --output_list_file data/dataset_train/ablation_${VARIANT}_pose_s2_pred_train.txt
# repeat for test

# 6) Prepare Pose-S3 data (recomputes rotation targets from raw pose + Pose-S2 output)
python ablation_test/scripts/prepare_pose_s3_data.py \
  --variant $VARIANT \
  --raw_list_file data/dataset_train/train_pose.txt \
  --pose_s2_list_file data/dataset_train/ablation_${VARIANT}_pose_s2_pred_train.txt \
  --output_dir data/dataset_train/ablation_${VARIANT}_pose_s3_data_train \
  --output_list_file data/dataset_train/ablation_${VARIANT}_pose_s3_data_train.txt
# repeat for test

# 7) Pose-S3
python ablation_test/scripts/train_pose_s3.py \
  --variant $VARIANT --branch all \
  --train_list_file data/dataset_train/ablation_${VARIANT}_pose_s3_data_train.txt \
  --test_list_file data/dataset_train/ablation_${VARIANT}_pose_s3_data_test.txt \
  --save_dir $CKPT/pose_s3

# 8) Precompute Pose-S3 outputs (assembled [T,90], fusion-ready schema)
python ablation_test/scripts/precompute_pose_s3_predictions.py \
  --variant $VARIANT \
  --data_list_file data/dataset_train/ablation_${VARIANT}_pose_s3_data_train.txt \
  --checkpoint_root $CKPT/pose_s3 \
  --output_dir data/dataset_train/ablation_${VARIANT}_pose_s3_pred_train \
  --output_list_file data/dataset_train/ablation_${VARIANT}_pose_s3_pred_train.txt
# repeat for test

# 9) Fusion -- the ORIGINAL, unmodified script. Branch-count agnostic.
python scripts/train/train_pose_s3_rotation_fusion.py \
  --train_pred_list_file data/dataset_train/ablation_${VARIANT}_pose_s3_pred_train.txt \
  --test_pred_list_file data/dataset_train/ablation_${VARIANT}_pose_s3_pred_test.txt \
  --save_dir $CKPT/fusion
```

Run this twice (`VARIANT=three_branch`, then `VARIANT=single_branch`).

## Getting the Table IV numbers

`evaluate_ablation_variant.py` runs the full S1->S2->S3->fusion->pose
pipeline end to end on the held-out TotalCapture test set (same
`PoseEvaluator` formula, same reduced-pose reconstruction, as
`scripts/profile/profile_full_pipeline_fivebranch.py`) and prints the exact
row Table IV needs:

```bash
python ablation_test/scripts/evaluate_ablation_variant.py \
  --variant $VARIANT \
  --checkpoint_root checkpoints/retrained/ablation_$VARIANT
# --raw_list_file defaults to data/dataset_train/test.txt (the same 45
# TotalCapture sequences Table II/III/IV already use)
```

Optional `--export_per_sequence <path.json>` writes per-sequence SIP/Ang/
Pos/Mesh/Jitter, in case you also want a Wilcoxon/bootstrap comparison for
a variant (see `scripts/analysis/statistical_significance.py`).

**Already validated end-to-end**, not just syntax-checked: I trained both
variants for 1 epoch on a 2-sequence subset (throwaway checkpoints, not
committed anywhere) and ran the full evaluate script against them --
the whole S1->S2->S3->fusion->pose->metrics chain executed without errors
and printed a well-formed Table IV row for both `three_branch` and
`single_branch`. The numbers were meaningless (1-epoch/2-sequence
checkpoints), which is expected and irrelevant -- what mattered was
catching integration bugs now instead of after a real multi-hour training
run on the server. One bug was caught and fixed this way (a std-aggregation
mismatch in the summary printer). Every script in this folder is ready to
run for real as soon as the server is reachable.
