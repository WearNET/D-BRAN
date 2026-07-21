"""
Centralized project paths for D-BRAN.

All project scripts should obtain data, checkpoint, result, and manifest
paths from this module.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


def add_project_root_to_path() -> Path:
    """
    Add the D-BRAN repository root to sys.path and return it.

    This function is kept for compatibility with scripts that may need to
    initialize the repository root explicitly.
    """
    project_root_string = str(PROJECT_ROOT)

    if project_root_string not in sys.path:
        sys.path.insert(0, project_root_string)

    return PROJECT_ROOT


def project_path(*parts: str) -> Path:
    """
    Build an absolute path relative to the D-BRAN repository root.
    """
    return PROJECT_ROOT.joinpath(*parts)


def require_path(path: Path, description: str = "Required path") -> Path:
    """
    Return a path if it exists, otherwise raise FileNotFoundError.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    return path


# ---------------------------------------------------------------------------
# Main directories
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
DATASET_RAW_DIR = DATA_DIR / "dataset_raw"
DATASET_WORK_DIR = DATA_DIR / "dataset_work"
DATASET_TRAIN_DIR = DATA_DIR / "dataset_train"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"


# ---------------------------------------------------------------------------
# Base model files
# ---------------------------------------------------------------------------

SMPL_MODEL_FILE = (
    DATA_DIR / "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
)

TRANSPOSE_WEIGHTS_FILE = (
    CHECKPOINTS_DIR / "transpose_original" / "weights.pt"
)


# ---------------------------------------------------------------------------
# Final D-BRAN checkpoints
# ---------------------------------------------------------------------------

POSE_S1_CHECKPOINTS_DIR = (
    CHECKPOINTS_DIR / "dbran_pose_s1_5branch_32h"
)

POSE_S2_CHECKPOINTS_DIR = (
    CHECKPOINTS_DIR / "dbran_pose_s2_5branch_16h"
)

POSE_S3_CHECKPOINTS_DIR = (
    CHECKPOINTS_DIR / "dbran_pose_s3_5branch_16h"
)

POSE_S3_FUSION_CHECKPOINT = (
    CHECKPOINTS_DIR
    / "dbran_pose_s3_fusion_16h"
    / "best_pose_s3_five_branch_rotation_fusion.pth"
)

# New training runs should not overwrite the final checkpoints.
RETRAINED_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "retrained"


# ---------------------------------------------------------------------------
# Base sequence manifests
# ---------------------------------------------------------------------------

TRAIN_POSE_LIST = DATASET_TRAIN_DIR / "train_pose.txt"
TEST_POSE_LIST = DATASET_TRAIN_DIR / "test.txt"


# ---------------------------------------------------------------------------
# PoseS1 manifests
# ---------------------------------------------------------------------------

POSE_S1_GT_TRAIN_LIST = (
    DATASET_TRAIN_DIR / "train_pose_s1_gt_no_classifier.txt"
)

POSE_S1_GT_TEST_LIST = (
    DATASET_TRAIN_DIR / "test_pose_s1_gt_no_classifier.txt"
)

POSE_S1_PRED_TRAIN_LIST = (
    DATASET_TRAIN_DIR / "train_pose_s1_pred_32h.txt"
)

POSE_S1_PRED_TEST_LIST = (
    DATASET_TRAIN_DIR / "test_pose_s1_pred_32h.txt"
)


# ---------------------------------------------------------------------------
# PoseS2 manifests
# ---------------------------------------------------------------------------

POSE_S2_GT_TRAIN_LIST = (
    DATASET_TRAIN_DIR / "train_pose_s2_gt.txt"
)

POSE_S2_GT_TEST_LIST = (
    DATASET_TRAIN_DIR / "test_pose_s2_gt.txt"
)

POSE_S2_TRAIN_LIST = (
    DATASET_TRAIN_DIR / "train_pose_s2_full_distributed.txt"
)

POSE_S2_TEST_LIST = (
    DATASET_TRAIN_DIR / "test_pose_s2_full_distributed.txt"
)

POSE_S2_PRED_TRAIN_LIST = (
    DATASET_TRAIN_DIR
    / "train_pose_s2_full_distributed_16h_pred.txt"
)

POSE_S2_PRED_TEST_LIST = (
    DATASET_TRAIN_DIR
    / "test_pose_s2_full_distributed_16h_pred.txt"
)


# ---------------------------------------------------------------------------
# PoseS3 manifests
# ---------------------------------------------------------------------------

POSE_S3_TRAIN_LIST = (
    DATASET_TRAIN_DIR / "train_pose_s3_five_branch_16h.txt"
)

POSE_S3_TEST_LIST = (
    DATASET_TRAIN_DIR / "test_pose_s3_five_branch_16h.txt"
)

POSE_S3_PRED_TRAIN_LIST = (
    DATASET_TRAIN_DIR
    / "train_pose_s3_five_branch_region_pred_16h.txt"
)

POSE_S3_PRED_TEST_LIST = (
    DATASET_TRAIN_DIR
    / "test_pose_s3_five_branch_region_pred_16h.txt"
)
