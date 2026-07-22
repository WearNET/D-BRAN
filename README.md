# D-BRAN

[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](#project-status)
[![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/task-sparse--IMU%20pose%20estimation-blue)](#overview)

**D-BRAN** is a distributed bidirectional recurrent architecture for full-body human pose estimation from six sparse inertial measurement units (IMUs).

The pose pipeline is divided into five anatomical branches that share the root IMU as a common reference. The current checkpoint set contains the final Pose-S1, Pose-S2, Pose-S3, and learned rotation-fusion models, together with the original TransPose translation branches.

> This repository currently represents the validated research implementation. The final evaluation entry point, Xsens–Unity demonstration, installation requirements, and documentation will continue to be consolidated.

---

## Overview

D-BRAN estimates full-body pose using six IMUs placed on:

- left forearm;
- right forearm;
- left lower leg;
- right lower leg;
- head;
- root/pelvis.

The architecture preserves the staged pose-estimation strategy while distributing the computation across anatomical branches.

```text
Six sparse IMUs
      │
      ├── shared root reference
      │
      ├── left-leg branch
      ├── right-leg branch
      ├── trunk-head branch
      ├── left-arm branch
      └── right-arm branch
      │
      ▼
Pose-S1: distributed leaf-joint position estimation
      │
      ▼
Pose-S2: distributed full-joint position estimation
      │
      ▼
Pose-S3: distributed reduced-joint rotation estimation
      │
      ▼
Learned residual rotation fusion
      │
      ▼
Full SMPL pose
```

The original TransPose `Trans-B1` and `Trans-B2` modules are retained for translation estimation.

---

## Final architecture

| Stage | Organization | Hidden size | Output |
|---|---:|---:|---|
| Pose-S1 | 5 distributed branches | 32 | Leaf-joint positions |
| Pose-S2 | 5 distributed branches | 16 | Full-joint positions by anatomical branch |
| Pose-S3 | 5 distributed branches | 16 | Reduced-joint 6D rotations by anatomical branch |
| Rotation fusion | Learned residual fusion | 16 | Refined 90-dimensional rotation representation |
| Translation | Original Trans-B1 and Trans-B2 | Original configuration | Root translation |

Pose-S1 uses one local IMU and the shared root IMU for each branch. Pose-S2 and Pose-S3 continue the distributed processing with branch-specific joint subsets. The fusion network refines the assembled Pose-S3 output before conversion to the full SMPL pose.

### Parameter count

The current full-pipeline profiler reports:

| Pipeline | Parameters | FP32 parameter size |
|---|---:|---:|
| Original TransPose full pipeline | 4,798,771 | 18.306 MB |
| D-BRAN pose pipeline + original Trans-B1/B2 | 1,603,357 | 6.116 MB |

The resulting parameter ratio is **0.3341**, corresponding to approximately **66.6% fewer parameters** than the original full pipeline.

---

## Repository structure

```text
D-BRAN/
├── articulate/                     # SMPL, kinematics, math, and evaluation utilities
├── checkpoints/
│   ├── transpose_original/
│   ├── dbran_pose_s1_5branch_32h/
│   ├── dbran_pose_s2_5branch_16h/
│   ├── dbran_pose_s3_5branch_16h/
│   └── dbran_pose_s3_fusion_16h/
├── data/
│   └── README.md
├── dbran/
│   └── __init__.py
├── scripts/
│   ├── baseline/                   # Original TransPose evaluation
│   ├── data/                       # Dataset and stage-data preparation
│   ├── figures/                    # Paper figure generation
│   ├── precompute/                 # Intermediate stage predictions
│   ├── profile/                    # Full-pipeline and FLOP profiling
│   └── train/                      # Pose-S1, Pose-S2, Pose-S3, and fusion training
├── config.py
├── main_path.py                    # Centralized repository paths
├── net.py                          # Original TransPose network and translation modules
├── utils.py
├── requirements.txt
└── README.md
```

All scripts under `scripts/` include a project-root bootstrap and use `main_path.py` as the centralized source for repository paths. They can therefore be executed directly with the usual path syntax:

```bash
python scripts/<group>/<script>.py
```

---

## Checkpoints

The repository includes the validated checkpoint set:

```text
checkpoints/
├── transpose_original/
│   └── weights.pt
├── dbran_pose_s1_5branch_32h/
│   ├── left_leg/
│   ├── right_leg/
│   ├── head/
│   ├── left_arm/
│   └── right_arm/
├── dbran_pose_s2_5branch_16h/
│   ├── left_leg/
│   ├── right_leg/
│   ├── trunk_head/
│   ├── left_arm/
│   └── right_arm/
├── dbran_pose_s3_5branch_16h/
│   ├── left_leg/
│   ├── right_leg/
│   ├── trunk_head/
│   ├── left_arm/
│   └── right_arm/
└── dbran_pose_s3_fusion_16h/
    └── best_pose_s3_five_branch_rotation_fusion.pth
```

New training runs are directed to `checkpoints/retrained/` so the validated checkpoints are not overwritten.

---

## Data

The datasets and generated training tensors are not tracked in Git because of their size and licensing conditions.

The current workflow uses:

- **AMASS** for synthetic training data;
- **DIP-IMU** for training and validation data;
- **TotalCapture** for evaluation.

Expected local directories include:

```text
data/
├── dataset_raw/
├── dataset_work/
├── dataset_train/
└── basicmodel_m_lbs_10_207_0_v1.0.0.pkl
```

The SMPL model file is not included in this repository. Place it at:

```text
data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl
```

The generated manifests and intermediate tensors used by the staged training workflow belong under:

```text
data/dataset_train/
```

See [`data/README.md`](data/README.md) for the current data-directory notes.

---

## Environment

The current implementation has been validated inside the existing Conda environment used for TransPose:

```bash
conda activate TransPose
```

`requirements.txt` is currently a placeholder. It will be completed after the final evaluation and Xsens–Unity demo dependencies are frozen.

At minimum, the current scripts require PyTorch and the scientific Python packages imported by the training, evaluation, profiling, and visualization scripts.

---

## Evaluation and profiling

### Smoke test

The following command loads every checkpoint and evaluates one sequence:

```bash
python scripts/profile/profile_full_pipeline_fivebranch.py \
  --raw_list_file data/dataset_train/test.txt \
  --max_sequences 1 \
  --profile_repeat 1
```

This verifies:

- original TransPose checkpoint loading;
- all five Pose-S1 branches;
- all five Pose-S2 branches;
- all five Pose-S3 branches;
- learned rotation fusion;
- original Trans-B1 and Trans-B2;
- offline pose evaluation;
- throughput and CUDA-memory profiling.

### Full offline evaluation

```bash
python scripts/profile/profile_full_pipeline_fivebranch.py \
  --raw_list_file data/dataset_train/test.txt \
  --profile_repeat 1
```

### Online protocol

```bash
python scripts/profile/profile_full_pipeline_fivebranch.py \
  --raw_list_file data/dataset_train/test.txt \
  --evaluate_online \
  --profile_repeat 1
```

The profiler reports:

- SIP error;
- angular error;
- positional error;
- mesh error;
- jitter;
- parameter count;
- time per frame;
- throughput;
- peak CUDA memory.

The optional robustness hook is not part of the final pipeline. A message indicating that `robustness_full_pipeline_hook` is unavailable can be ignored during standard evaluation.

---

## Training and data preparation

The staged workflow is preserved under `scripts/`.

### Pose-S1

```text
scripts/data/prepare_pose_s1_gt.py
scripts/train/train_pose_s1_distributed.py
scripts/precompute/precompute_pose_s1_predictions.py
```

### Pose-S2

```text
scripts/data/prepare_pose_s2_gt.py
scripts/data/prepare_pose_s2_full_distributed.py
scripts/train/train_pose_s2_full_distributed.py
scripts/precompute/precompute_pose_s2_full_distributed_outputs.py
```

### Pose-S3 and fusion

```text
scripts/data/prepare_pose_s3_five_branch.py
scripts/train/train_pose_s3_region.py
scripts/precompute/precompute_pose_s3_region_outputs.py
scripts/train/train_pose_s3_rotation_fusion.py
```

Run any script with `--help` to inspect its current arguments:

```bash
python scripts/train/train_pose_s2_full_distributed.py --help
```

---

## Project status

- [x] Final Pose-S1 checkpoint set migrated
- [x] Final Pose-S2 checkpoint set migrated
- [x] Final Pose-S3 checkpoint set migrated
- [x] Final learned fusion checkpoint migrated
- [x] Original TransPose translation checkpoint migrated
- [x] Centralized repository paths implemented
- [x] End-to-end CUDA pipeline validated
- [x] Training and preprocessing scripts preserved
- [ ] Dedicated `evaluate_dbran.py` entry point
- [ ] Xsens–Unity real-time demo
- [ ] Final `requirements.txt`
- [ ] Complete reproducibility instructions
- [ ] Final paper citation and release documentation

---

## Planned work

The next repository updates will focus on:

1. a clean root-level D-BRAN evaluation script;
2. real-time inference with Xsens MTw sensors;
3. Unity communication and avatar visualization;
4. finalized dependency installation;
5. consolidated dataset preparation instructions;
6. publication results and citation information.

---

## Acknowledgments

This work builds on the TransPose sparse-IMU pose-estimation pipeline and its supporting articulation and SMPL utilities.

The original TransPose pose stages are replaced by the distributed D-BRAN architecture, while the original translation branches are retained in the current full pipeline.

---

## License and citation

License information will be added after the terms of the upstream source code, datasets, SMPL assets, and derived checkpoint distribution have been fully reviewed.

A citation entry will be added after publication information is available.
