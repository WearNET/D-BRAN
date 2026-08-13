"""
Paired significance test between D-BRAN and the centralized TransPose
reference over the held-out TotalCapture test sequences.

Input is the per-sequence JSON produced by
    scripts/profile/profile_full_pipeline_fivebranch.py --export_per_sequence ...

Computes, for a chosen metric column (default: offline SIP error):
    - Wilcoxon signed-rank test (paired, two-sided) on (distributed - original).
    - Bootstrap 95% confidence interval on the median paired difference.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_median_ci(differences: np.ndarray, num_resamples: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = differences.shape[0]
    resample_medians = np.empty(num_resamples, dtype=float)

    for i in range(num_resamples):
        sample = differences[rng.integers(0, n, size=n)]
        resample_medians[i] = np.median(sample)

    lower = np.percentile(resample_medians, 100 * (alpha / 2))
    upper = np.percentile(resample_medians, 100 * (1 - alpha / 2))
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_sequence_json", required=True)
    parser.add_argument("--metric_prefix", default="sip", help="Column prefix, e.g. 'sip' -> sip_orig/sip_dist")
    parser.add_argument("--num_resamples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.per_sequence_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    orig_key = f"{args.metric_prefix}_orig"
    dist_key = f"{args.metric_prefix}_dist"

    original = np.asarray([r[orig_key] for r in records], dtype=float)
    distributed = np.asarray([r[dist_key] for r in records], dtype=float)
    differences = distributed - original

    n = len(records)
    print(f"Sequences: {n}")
    print(f"Metric:    {args.metric_prefix} ({dist_key} - {orig_key})")
    print(f"Original   mean +/- std: {original.mean():.4f} +/- {original.std(ddof=0):.4f}")
    print(f"Distributed mean +/- std: {distributed.mean():.4f} +/- {distributed.std(ddof=0):.4f}")
    print(f"Paired difference median: {np.median(differences):.4f}")
    print(f"Paired difference mean:   {differences.mean():.4f} +/- {differences.std(ddof=0):.4f}")

    if np.all(differences == 0):
        print("All paired differences are zero; Wilcoxon test is undefined.")
        return

    statistic, p_value = wilcoxon(distributed, original, alternative="two-sided", zero_method="wilcox")
    print(f"\nWilcoxon signed-rank test (two-sided): W = {statistic:.4f}, p = {p_value:.6g}")

    lower, upper = bootstrap_median_ci(differences, args.num_resamples, args.seed)
    print(f"Bootstrap 95% CI on median difference ({args.num_resamples} resamples, seed={args.seed}): "
          f"[{lower:.4f}, {upper:.4f}] deg")


if __name__ == "__main__":
    main()
