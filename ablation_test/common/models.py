"""
Model classes for the ablation variants.

These are structurally identical to the originals in
scripts/train/train_pose_s1_distributed.py,
scripts/train/train_pose_s2_full_distributed.py,
scripts/train/train_pose_s3_region.py, and
scripts/train/train_pose_s3_rotation_fusion.py -- only input_dim/output_dim
change per branch. Copied here (rather than imported) so ablation_test/
stays self-contained and easy to remove.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class BranchRNN(nn.Module):
    """
    Generic fc_in -> BiLSTM -> fc_out branch network, used for Pose-S1,
    Pose-S2, and Pose-S3 stages alike (they only differ in dims/defaults).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_dim: int,
        rnn_hidden: int,
        rnn_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.LSTM(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        if not isinstance(sequence, PackedSequence):
            raise RuntimeError("BranchRNN expects a PackedSequence.")

        data = self.dropout(sequence.data)
        data = torch.relu(self.fc_in(data))

        packed = PackedSequence(
            data,
            sequence.batch_sizes,
            sequence.sorted_indices,
            sequence.unsorted_indices,
        )
        output, _ = self.rnn(packed)
        prediction = self.fc_out(output.data)

        return PackedSequence(
            prediction,
            output.batch_sizes,
            output.sorted_indices,
            output.unsorted_indices,
        )


class FusionRNN(nn.Module):
    """Same structure as PoseS3FiveBranchFusionNet -- reused unchanged."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 90,
        proj_dim: int = 16,
        rnn_hidden: int = 16,
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.LSTM(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        if not isinstance(sequence, PackedSequence):
            raise RuntimeError("FusionRNN expects a PackedSequence.")

        data = self.dropout(sequence.data)
        data = torch.relu(self.fc_in(data))

        packed = PackedSequence(
            data,
            sequence.batch_sizes,
            sequence.sorted_indices,
            sequence.unsorted_indices,
        )
        output, _ = self.rnn(packed)
        delta = self.fc_out(output.data)

        return PackedSequence(
            delta,
            output.batch_sizes,
            output.sorted_indices,
            output.unsorted_indices,
        )
