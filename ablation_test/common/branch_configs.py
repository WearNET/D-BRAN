"""
Shared branch/sensor/joint group definitions for the ablation architecture
variants ("three_branch" and "single_branch") required by the paper's
Table IV: "Three-branch partition (legs, trunk-head, arms)" and
"Single branch, same width".

These groupings reuse the exact joint sets already used elsewhere in the
D-BRAN repo:
    - legs / arms / trunk_head match LOWER_BODY_JOINTS / UPPER_BODY_JOINTS /
      TRUNK_HEAD_JOINTS from scripts/data/prepare_pose_s2_gt.py.
    - The five-branch reduced_joints (used to derive the three-branch
      groupings) match scripts/train/train_pose_s3_region.py's BRANCH_CONFIG.

Both variants train from scratch on the SAME protocol as the original
five-branch D-BRAN (AMASS + DIP-IMU, no custom OptiTrack/Xsens data), so
Table IV stays a clean architecture-only comparison.

Every stage's per-branch config carries "input_dim"/"output_dim" purely as
a sanity-check value -- the actual dimension always comes from the tensors
built at runtime.
"""

from __future__ import annotations

# ------------------------------------------------------------
# Sensor indices (consistent across the whole D-BRAN codebase)
#   0 = left arm, 1 = right arm, 2 = left leg, 3 = right leg,
#   4 = head, 5 = root
# ------------------------------------------------------------
LEFT_ARM_SENSOR_IDX = 0
RIGHT_ARM_SENSOR_IDX = 1
LEFT_LEG_SENSOR_IDX = 2
RIGHT_LEG_SENSOR_IDX = 3
HEAD_SENSOR_IDX = 4
ROOT_SENSOR_IDX = 5

# All 15 reduced (6D-rotation) joints, union of the five original branches'
# reduced_joints -- matches config.py's joint_set.reduced.
REDUCED_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]

# All 23 non-root SMPL joints (full position target set).
ALL_JOINTS = list(range(1, 24))


# ==============================================================
# THREE-BRANCH: legs, trunk_head, arms
# ==============================================================
THREE_BRANCH_ORDER = ["legs", "trunk_head", "arms"]

THREE_BRANCH_S1_CONFIG = {
    "legs": {
        "local_sensor_indices": [LEFT_LEG_SENSOR_IDX, RIGHT_LEG_SENSOR_IDX],
        "leaf_gt_keys": ["left_leg_p_gt", "right_leg_p_gt"],
        "input_dim": 36,
        "output_dim": 6,
    },
    "trunk_head": {
        "local_sensor_indices": [HEAD_SENSOR_IDX],
        "leaf_gt_keys": ["head_p_gt"],
        "input_dim": 24,
        "output_dim": 3,
    },
    "arms": {
        "local_sensor_indices": [LEFT_ARM_SENSOR_IDX, RIGHT_ARM_SENSOR_IDX],
        "leaf_gt_keys": ["left_arm_p_gt", "right_arm_p_gt"],
        "input_dim": 36,
        "output_dim": 6,
    },
}

THREE_BRANCH_S2_CONFIG = {
    "legs": {
        "local_sensor_indices": [LEFT_LEG_SENSOR_IDX, RIGHT_LEG_SENSOR_IDX],
        "joints": [1, 2, 4, 5, 7, 8, 10, 11],
        "input_dim": 42,
        "output_dim": 24,
    },
    "trunk_head": {
        "local_sensor_indices": [HEAD_SENSOR_IDX],
        "joints": [3, 6, 9, 12, 15],
        "input_dim": 27,
        "output_dim": 15,
    },
    "arms": {
        "local_sensor_indices": [LEFT_ARM_SENSOR_IDX, RIGHT_ARM_SENSOR_IDX],
        "joints": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
        "input_dim": 42,
        "output_dim": 30,
    },
}

THREE_BRANCH_S3_CONFIG = {
    "legs": {
        "sensor_indices": [ROOT_SENSOR_IDX, LEFT_LEG_SENSOR_IDX, RIGHT_LEG_SENSOR_IDX],
        "position_joints": [1, 2, 4, 5, 7, 8, 10, 11],
        "reduced_joints": [1, 2, 4, 5],
        "input_dim": 60,
        "output_dim": 24,
    },
    "trunk_head": {
        "sensor_indices": [ROOT_SENSOR_IDX, HEAD_SENSOR_IDX],
        "position_joints": [3, 6, 9, 12, 15],
        "reduced_joints": [3, 6, 9, 12, 15],
        "input_dim": 39,
        "output_dim": 30,
    },
    "arms": {
        "sensor_indices": [ROOT_SENSOR_IDX, LEFT_ARM_SENSOR_IDX, RIGHT_ARM_SENSOR_IDX],
        "position_joints": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
        "reduced_joints": [13, 14, 16, 17, 18, 19],
        "input_dim": 66,
        "output_dim": 36,
    },
}


# ==============================================================
# SINGLE BRANCH: one network per stage, same recurrent width as
# the original per-branch networks (proj_dim/rnn_hidden unchanged),
# covering every sensor / joint at once.
# ==============================================================
SINGLE_BRANCH_ORDER = ["single"]

_SINGLE_LOCAL_SENSOR_ORDER = [
    LEFT_LEG_SENSOR_IDX,
    RIGHT_LEG_SENSOR_IDX,
    HEAD_SENSOR_IDX,
    LEFT_ARM_SENSOR_IDX,
    RIGHT_ARM_SENSOR_IDX,
]

SINGLE_BRANCH_S1_CONFIG = {
    "single": {
        "local_sensor_indices": _SINGLE_LOCAL_SENSOR_ORDER,
        "leaf_gt_keys": [
            "left_leg_p_gt",
            "right_leg_p_gt",
            "head_p_gt",
            "left_arm_p_gt",
            "right_arm_p_gt",
        ],
        "input_dim": 72,
        "output_dim": 15,
    },
}

SINGLE_BRANCH_S2_CONFIG = {
    "single": {
        "local_sensor_indices": _SINGLE_LOCAL_SENSOR_ORDER,
        "joints": ALL_JOINTS,
        "input_dim": 87,
        "output_dim": 69,
    },
}

SINGLE_BRANCH_S3_CONFIG = {
    "single": {
        "sensor_indices": [ROOT_SENSOR_IDX] + _SINGLE_LOCAL_SENSOR_ORDER,
        "position_joints": ALL_JOINTS,
        "reduced_joints": REDUCED_JOINTS,
        "input_dim": 141,
        "output_dim": 90,
    },
}


# ==============================================================
# Variant registry
# ==============================================================
VARIANTS = {
    "three_branch": {
        "order": THREE_BRANCH_ORDER,
        "s1": THREE_BRANCH_S1_CONFIG,
        "s2": THREE_BRANCH_S2_CONFIG,
        "s3": THREE_BRANCH_S3_CONFIG,
    },
    "single_branch": {
        "order": SINGLE_BRANCH_ORDER,
        "s1": SINGLE_BRANCH_S1_CONFIG,
        "s2": SINGLE_BRANCH_S2_CONFIG,
        "s3": SINGLE_BRANCH_S3_CONFIG,
    },
}


def get_variant(variant_name: str):
    if variant_name not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant_name}'. Choices: {list(VARIANTS.keys())}"
        )
    return VARIANTS[variant_name]


def validate_variant(variant_name: str) -> None:
    """
    Sanity-check that a variant's S1/S2/S3 branches partition the sensors
    / joints / reduced joints exactly once, mirroring the five-branch
    design's own validate_branch_partition() check.
    """
    variant = get_variant(variant_name)
    order = variant["order"]

    all_local_sensors = []
    for branch in order:
        all_local_sensors.extend(variant["s1"][branch]["local_sensor_indices"])
    expected_sensors = sorted(
        [LEFT_ARM_SENSOR_IDX, RIGHT_ARM_SENSOR_IDX, LEFT_LEG_SENSOR_IDX,
         RIGHT_LEG_SENSOR_IDX, HEAD_SENSOR_IDX]
    )
    if sorted(all_local_sensors) != expected_sensors:
        raise RuntimeError(
            f"[{variant_name}] S1 local sensors must partition the five "
            f"peripheral sensors exactly once. Expected {expected_sensors}, "
            f"got {sorted(all_local_sensors)}"
        )

    all_s2_joints = []
    for branch in order:
        all_s2_joints.extend(variant["s2"][branch]["joints"])
    if sorted(all_s2_joints) != ALL_JOINTS:
        raise RuntimeError(
            f"[{variant_name}] S2 joints must partition all 23 joints "
            f"exactly once. Got {sorted(all_s2_joints)}"
        )

    all_reduced = []
    for branch in order:
        all_reduced.extend(variant["s3"][branch]["reduced_joints"])
    if sorted(all_reduced) != sorted(REDUCED_JOINTS):
        raise RuntimeError(
            f"[{variant_name}] S3 reduced_joints must partition "
            f"REDUCED_JOINTS exactly once. Got {sorted(all_reduced)}"
        )


if __name__ == "__main__":
    for name in VARIANTS:
        validate_variant(name)
        print(f"[ok] {name} partitions validated.")
