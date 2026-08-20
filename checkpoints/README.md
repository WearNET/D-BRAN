# Checkpoints

This directory contains the validated model weights used by the current D-BRAN pipeline.

Unlike the datasets, the final checkpoints are tracked in this repository so the trained architecture can be evaluated without retraining every stage.

## Directory structure

```text
checkpoints/
├── README.md
├── transpose_original/
│   └── weights.pt
├── dbran_pose_s1/
├── dbran_pose_s2/
├── dbran_pose_s3/
├── dbran_pose_s3_fusion/
└── retrained/                      # New training runs (gitignored, see below)
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
checkpoints/dbran_pose_s1/
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
checkpoints/dbran_pose_s2/
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
checkpoints/dbran_pose_s3/
```

Pose-S3 estimates reduced-joint 6D rotations through five anatomical branches.

The final Pose-S3 configuration uses a recurrent hidden size of 16.

## Learned rotation fusion

```text
checkpoints/dbran_pose_s3_fusion/
```

Checkpoint:

```text
best_pose_s3_fusion.pth
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

## Retraining

New training runs should be saved under:

```text
checkpoints/retrained/
```

This directory is excluded from Git so experimental runs do not overwrite or become confused with the validated checkpoint set.

## Smoke test

The complete checkpoint set can be tested with:

```bash
python scripts/profile/profile_full_pipeline.py \
  --raw_list_file data/dataset_train/test.txt \
  --max_sequences 1 \
  --profile_repeat 1
```

`--max_sequences 1` loads every checkpoint and evaluates a single sequence, purely to confirm all branches and the fusion network load and run — not a representative evaluation. It verifies all D-BRAN pose branches, the fusion network, and the original TransPose translation modules.

Each training script's argparse defaults reproduce the original D-BRAN study exactly (same hyperparameters, same data splits), so running a script without overriding its defaults is the correct way to redo a stage identically. The exception is the output path: Pose-S1 and Pose-S2 training default `--save_dir` to `checkpoints/retrained/`, so a default run never touches the validated checkpoints above. Pose-S3 and fusion training require an explicit `--save_dir`; pointing it at `checkpoints/dbran_pose_s3/` or `checkpoints/dbran_pose_s3_fusion/` overwrites the validated checkpoint currently loaded by the smoke test, so always redirect it to a new or `checkpoints/retrained/`-style directory when reproducing the study.
