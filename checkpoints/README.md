# Checkpoints

This directory contains the validated model weights used by the current D-BRAN pipeline.

Unlike the datasets, the final checkpoints are tracked in this repository so the trained architecture can be evaluated without retraining every stage.

## Directory structure

```text
checkpoints/
├── README.md
├── transpose_original/
│   └── weights.pt
├── dbran_pose_s1_5branch_32h/
├── dbran_pose_s2_5branch_16h/
├── dbran_pose_s3_5branch_16h/
├── dbran_pose_s3_fusion_16h/
└── retrained/                      # Fine-tuned checkpoints (gitignored, see below)
```

## Original TransPose checkpoint

```text
checkpoints/transpose_original/weights.pt
```

This checkpoint contains the original TransPose model parameters.

In the current D-BRAN pipeline:

- the original TransPose model is retained as the centralized baseline;
- the original `Trans-B1` and `Trans-B2` modules are retained for translation estimation;
- the original centralized pose stages are replaced by the distributed D-BRAN pose architecture.

## D-BRAN Pose-S1

```text
checkpoints/dbran_pose_s1_5branch_32h/
```

Pose-S1 estimates five leaf-joint positions using independent branches based on the shared root IMU and one local IMU.

Branches:

```text
left_leg
right_leg
head
left_arm
right_arm
```

The final Pose-S1 configuration uses a recurrent hidden size of 32 and does not use the discarded sector classifier.

## D-BRAN Pose-S2

```text
checkpoints/dbran_pose_s2_5branch_16h/
```

Pose-S2 estimates full-joint positions through five anatomical branches:

```text
left_leg
right_leg
trunk_head
left_arm
right_arm
```

The final Pose-S2 configuration uses a recurrent hidden size of 16.

## D-BRAN Pose-S3

```text
checkpoints/dbran_pose_s3_5branch_16h/
```

Pose-S3 estimates reduced-joint 6D rotations through five anatomical branches.

The checkpoint filenames preserve the original development name `region`, but the final system is described as a distributed five-branch architecture.

The final Pose-S3 configuration uses a recurrent hidden size of 16.

## Learned rotation fusion

```text
checkpoints/dbran_pose_s3_fusion_16h/
```

Checkpoint:

```text
best_pose_s3_five_branch_rotation_fusion.pth
```

The fusion network refines the assembled Pose-S3 output using a learned residual correction.

Current configuration:

```text
input_dim: 90
output_dim: 90
hidden_size: 16
use_pose_s2_position: False
```

## Centralized checkpoint paths

All checkpoint locations are defined in:

```text
main_path.py
```

Relevant constants:

```python
TRANSPOSE_WEIGHTS_FILE
POSE_S1_CHECKPOINTS_DIR
POSE_S2_CHECKPOINTS_DIR
POSE_S3_CHECKPOINTS_DIR
POSE_S3_FUSION_CHECKPOINT
```

Scripts should use these constants instead of hard-coded absolute paths.

## Retraining and fine-tuning

New training runs should be saved under:

```text
checkpoints/retrained/
```

This directory is excluded from Git (`.gitignore`) so experimental and fine-tuned runs never overwrite or get confused with the validated checkpoint set above. Its structure mirrors the validated one:

```text
checkpoints/retrained/
├── pose_s1_5branch_32h/<target>/best_pose_s1_no_classifier_<target>.pth
├── pose_s2_full_distributed/<target>/best_pose_s2_full_distributed_<target>.pth
├── pose_s3_5branch_16h/<target>/best_pose_s3_five_branch_region_<target>.pth
└── pose_s3_fusion_16h/best_pose_s3_five_branch_rotation_fusion.pth
```

Every training script accepts a flag to continue from an existing checkpoint instead of training from scratch — see the [top-level README's fine-tuning section](../README.md#fine-tuning-on-custom-captures-optitrack--xsens) for the full workflow (capturing data, generating ground truth, and chaining the four stages). Summary of the flags:

| Script | Flag | Per-branch path built how |
|---|---|---|
| `train_pose_s1_distributed.py` | `--pretrained_checkpoint <file>` | Manual — pass each branch's own file explicitly |
| `train_pose_s2_full_distributed.py` | `--pretrained_checkpoint_dir <root>` | Automatic from `--target` |
| `train_pose_s3_region.py` | `--pretrained_checkpoint_dir <root>` | Automatic from `--target` |
| `train_pose_s3_rotation_fusion.py` | `--pretrained_checkpoint <file>` | N/A — single unified model |

Pose-S1's flag takes a single file because that was the original design; **be careful not to pass the same file for all 5 branches when fine-tuning Pose-S1** — since every branch shares the same architecture, loading one branch's weights into another loads without any error, silently starting that branch from the wrong pretrained weights. Pose-S2 and Pose-S3 use a directory instead specifically to make that mistake structurally impossible.

Architecture hyperparameters (`proj_dim`, `rnn_hidden`, `rnn_layers`, `dropout`) are always read from the checkpoint being loaded, overriding whatever was passed on the command line, so a fine-tuning run can never end up with a mismatched architecture.

## Validation

The complete checkpoint set can be tested with:

```bash
python scripts/profile/profile_full_pipeline_fivebranch.py \
  --raw_list_file data/dataset_train/test.txt \
  --max_sequences 1 \
  --profile_repeat 1
```

This verifies all D-BRAN pose branches, the fusion network, and the original TransPose translation modules.

To compare a fine-tuned checkpoint set against the validated originals on the standard benchmark, point all four `--*_root`/`--*_weights` flags at `checkpoints/retrained/...` in a second run and compare the "Distributed five-branch pose + original Trans-B1/B2" row between the two:

```bash
python scripts/profile/profile_full_pipeline_fivebranch.py \
  --raw_list_file data/dataset_train/test.txt \
  --distributed_s1_root checkpoints/retrained/pose_s1_5branch_32h \
  --pose_s2_full_root checkpoints/retrained/pose_s2_full_distributed \
  --pose_s3_five_branch_root checkpoints/retrained/pose_s3_5branch_16h \
  --pose_s3_fusion_weights checkpoints/retrained/pose_s3_fusion_16h/best_pose_s3_five_branch_rotation_fusion.pth \
  --evaluate_online \
  --profile_repeat 1
```

Only compare accuracy metrics (SIP/angular/positional/mesh/jitter error) across two separate runs — latency and throughput numbers are not reliable to compare *between* runs, since GPU/system load varies session to session (confirmed by the unchanged "Original full pipeline" row itself changing speed between runs that used identical weights). Latency is only meaningful compared *within* a single run.
