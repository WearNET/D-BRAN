# Data Directory

This directory contains the datasets, generated tensors, sequence manifests, and SMPL assets used by D-BRAN.

Large datasets and generated training files are not tracked in Git because of their size and licensing restrictions. They must be obtained separately and placed in the expected local directories.

## Directory structure

```text
data/
├── README.md
├── basicmodel_m_lbs_10_207_0_v1.0.0.pkl
├── dataset_raw/
│   ├── AMASS/
│   ├── DIP_IMU/
│   └── TotalCapture/
│       ├── DIP_recalculate/
│       └── official/
├── dataset_work/
│   ├── AMASS/
│   ├── DIP_IMU/
│   └── TotalCapture/
└── dataset_train/
```

## Raw datasets

D-BRAN currently uses:

- **AMASS** for synthetic sparse-IMU training data;
- **DIP-IMU** for real inertial training and validation data;
- **TotalCapture** for evaluation.

The datasets are not redistributed with this repository. Users are responsible for obtaining them from their official sources and complying with their respective licenses.

## SMPL model

The SMPL model file is not included in this repository.

Place the required model at:

```text
data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl
```

The path is configured centrally in `main_path.py`.

## Processed datasets

The base preprocessing script is:

```text
scripts/data/preprocess_base_datasets.py
```

Processed datasets are written under:

```text
data/dataset_work/
```

## Staged D-BRAN training data

The final training workflow uses intermediate outputs generated between stages.

### Pose-S1

```text
pose_s1_gt_train_no_classifier/
pose_s1_gt_test_no_classifier/
pose_s1_pred_train_32h/
pose_s1_pred_test_32h/
```

Manifests:

```text
train_pose_s1_gt_no_classifier.txt
test_pose_s1_gt_no_classifier.txt
train_pose_s1_pred_32h.txt
test_pose_s1_pred_32h.txt
```

### Pose-S2

```text
pose_s2_gt_train/
pose_s2_gt_test/
pose_s2_full_distributed_train/
pose_s2_full_distributed_test/
pose_s2_full_distributed_16h_pred_train/
pose_s2_full_distributed_16h_pred_test/
```

Manifests:

```text
train_pose_s2_gt.txt
test_pose_s2_gt.txt
train_pose_s2_full_distributed.txt
test_pose_s2_full_distributed.txt
train_pose_s2_full_distributed_16h_pred.txt
test_pose_s2_full_distributed_16h_pred.txt
```

### Pose-S3 and fusion

```text
pose_s3_five_branch_train_16h/
pose_s3_five_branch_test_16h/
pose_s3_five_branch_region_pred_train_16h/
pose_s3_five_branch_region_pred_test_16h/
```

Manifests:

```text
train_pose_s3_five_branch_16h.txt
test_pose_s3_five_branch_16h.txt
train_pose_s3_five_branch_region_pred_16h.txt
test_pose_s3_five_branch_region_pred_16h.txt
```

## Base sequence manifests

The main sequence lists are:

```text
data/dataset_train/train_pose.txt
data/dataset_train/test.txt
```

Example smoke test:

```bash
python scripts/profile/profile_full_pipeline_fivebranch.py \
  --raw_list_file data/dataset_train/test.txt \
  --max_sequences 1
```

## Important notes

- Dataset directories and generated tensors are excluded through `.gitignore`.
- Do not commit proprietary or license-restricted datasets.
- Generated `.pt` files may contain source-path metadata used to match sequences across stages.
- All central data paths are defined in `main_path.py`.
