"""
Generate paper-ready figures from the per-sequence CSV saved by
profile_full_pipeline_fivebranch_independent_with_csv.py.

Figures:
    Fig. 5: ECDF of offline mesh error.
    Fig. 6: ECDF of online jitter.
    Fig. 7: Computational trade-off relative to centralized TransPose.

Outputs are saved as both PDF (vector) and PNG.
"""

from __future__ import annotations

# BEGIN D-BRAN PROJECT BOOTSTRAP
import sys as _dbran_sys
from pathlib import Path as _DbranPath

_dbran_current_file = _DbranPath(__file__).resolve()

for _dbran_candidate in (
    _dbran_current_file.parent,
    *_dbran_current_file.parents,
):
    if (_dbran_candidate / "main_path.py").is_file():
        _dbran_root_string = str(_dbran_candidate)

        if _dbran_root_string not in _dbran_sys.path:
            _dbran_sys.path.insert(0, _dbran_root_string)

        break
else:
    raise RuntimeError(
        "Could not locate the D-BRAN project root from "
        f"{_dbran_current_file}"
    )

from main_path import PROJECT_ROOT
# END D-BRAN PROJECT BOOTSTRAP
from main_path import RESULTS_DIR



import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Global plotting style
# -----------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 12


# -----------------------------------------------------------------------------
# Pastel palette inspired by the architecture figure
# -----------------------------------------------------------------------------
COLOR_CENTRALIZED = "#7FA8E6"   # soft pastel blue
COLOR_DISTRIBUTED = "#F2A97F"   # soft pastel orange/peach

# Trade-off bars:
# Negative values are improvements/reductions, so they are green.
# Positive values are increases/costs, so they are red.
COLOR_BAR_GOOD = "#A8D5BA"      # pastel green
COLOR_BAR_BAD = "#F2A6A6"       # pastel red

COLOR_GRID = "#C8C8C8"



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_column(csv_file: str, column_name: str) -> np.ndarray:
    values = []

    with open(csv_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None or column_name not in reader.fieldnames:
            raise KeyError(
                f"Column '{column_name}' not found in {csv_file}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            value = row.get(column_name, "")
            if value == "":
                continue
            values.append(float(value))

    if len(values) == 0:
        raise RuntimeError(f"No numeric values found for column '{column_name}'.")

    return np.asarray(values, dtype=float)


def ecdf(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = np.sort(values)
    cumulative = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
    return values, cumulative


def extend_ecdf_to_common_max(
    x_original: np.ndarray,
    y_original: np.ndarray,
    x_distributed: np.ndarray,
    y_distributed: np.ndarray,
):
    """
    Extend both ECDF curves horizontally to the same maximum x-value.
    This avoids one curve appearing visually truncated.
    """
    x_max = max(float(x_original[-1]), float(x_distributed[-1]))

    if x_original[-1] < x_max:
        x_original = np.append(x_original, x_max)
        y_original = np.append(y_original, y_original[-1])

    if x_distributed[-1] < x_max:
        x_distributed = np.append(x_distributed, x_max)
        y_distributed = np.append(y_distributed, y_distributed[-1])

    return x_original, y_original, x_distributed, y_distributed


def save_current_figure(output_base: str) -> None:
    """
    Save a vector PDF and a high-resolution PNG preview.
    """
    plt.tight_layout(pad=0.25)

    plt.savefig(
        output_base + ".pdf",
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "Matplotlib",
        },
    )

    plt.savefig(
        output_base + ".png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()


def plot_ecdf(
    original_values: np.ndarray,
    distributed_values: np.ndarray,
    xlabel: str,
    output_base: str,
) -> None:
    x_original, y_original = ecdf(original_values)
    x_distributed, y_distributed = ecdf(distributed_values)

    x_original, y_original, x_distributed, y_distributed = extend_ecdf_to_common_max(
        x_original, y_original, x_distributed, y_distributed
    )

    plt.figure(figsize=(4.8, 3.2))

    plt.plot(
        x_original,
        y_original,
        linewidth=2.4,
        linestyle="-",
        color=COLOR_CENTRALIZED,
        label="Centralized TransPose",
    )

    plt.plot(
        x_distributed,
        y_distributed,
        linewidth=2.4,
        linestyle="-",
        color=COLOR_DISTRIBUTED,
        label="Distributed five-branch",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Cumulative fraction")

    plt.grid(True, linestyle="-", linewidth=0.7, color=COLOR_GRID, alpha=0.55)
    plt.legend(frameon=False, loc="lower right", handlelength=2.3)

    save_current_figure(output_base)


def plot_tradeoff(output_base: str) -> None:
    metrics = [
        "Full parameters",
        "Pose parameters",
        "FP32 model size",
        "Offline time/frame",
        "Offline CUDA memory",
        "Online CUDA memory",
        "Online latency",
    ]

    original = np.asarray([
        4_798_771,   # Full parameters
        3_537_198,   # Pose parameters
        18.306,      # FP32 model size MB
        0.062626,    # Offline time/frame ms
        210.893,     # Offline peak CUDA alloc MB
        93.509,      # Online peak CUDA alloc MB
        2.482485,    # Online latency ms
    ], dtype=float)

    distributed = np.asarray([
        1_603_357,
        341_784,
        6.116,
        0.043017,
        145.271,
        83.886,
        4.873927,
    ], dtype=float)

    percent_change = ((distributed / original) - 1.0) * 100.0
    y_positions = np.arange(len(metrics), dtype=float)

    bar_colors = [COLOR_BAR_BAD if value > 0 else COLOR_BAR_GOOD for value in percent_change]

    plt.figure(figsize=(7.0, 3.6))
    bars = plt.barh(
        y_positions,
        percent_change,
        color=bar_colors,
        edgecolor="none",
        height=0.75,
    )

    plt.axvline(0.0, linewidth=1.0, color="black")
    plt.yticks(y_positions, metrics)
    plt.xlabel("Change relative to centralized TransPose (%)")
    plt.grid(True, axis="x", linestyle="-", linewidth=0.7, color=COLOR_GRID, alpha=0.55)

    x_min = float(np.min(percent_change))
    x_max = float(np.max(percent_change))
    margin = max(4.0, 0.08 * (x_max - x_min))
    plt.xlim(x_min - margin * 1.8, x_max + margin * 1.4)

    for bar, value in zip(bars, percent_change):
        y = bar.get_y() + bar.get_height() / 2.0
        if value >= 0:
            plt.text(value + 1.0, y, f"{value:.1f}%", va="center", ha="left")
        else:
            plt.text(value - 1.0, y, f"{value:.1f}%", va="center", ha="right")

    save_current_figure(output_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        required=True,
        help="CSV generated by profile_full_pipeline_fivebranch_independent_with_csv.py",
    )
    parser.add_argument(
        "--output_dir",
        default=str(RESULTS_DIR / "paper_figures"),
        help="Directory where PDF and PNG figures will be saved.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    offline_mesh_original = read_csv_column(args.csv_file, "offline_original_mesh_cm")
    offline_mesh_distributed = read_csv_column(args.csv_file, "offline_distributed_mesh_cm")

    online_jitter_original = read_csv_column(args.csv_file, "online_original_jitter")
    online_jitter_distributed = read_csv_column(args.csv_file, "online_distributed_jitter")

    plot_ecdf(
        original_values=offline_mesh_original,
        distributed_values=offline_mesh_distributed,
        xlabel="Offline mesh error (cm)",
        output_base=os.path.join(args.output_dir, "fig5_offline_mesh_error_cdf"),
    )

    plot_ecdf(
        original_values=online_jitter_original,
        distributed_values=online_jitter_distributed,
        xlabel=r"Online jitter (100 m/s$^3$)",
        output_base=os.path.join(args.output_dir, "fig6_online_jitter_cdf"),
    )

    plot_tradeoff(
        output_base=os.path.join(args.output_dir, "fig7_computational_tradeoff"),
    )

    print(f"Saved paper figures to: {args.output_dir}")


if __name__ == "__main__":
    main()