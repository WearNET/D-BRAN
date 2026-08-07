"""Capture calibrated Xsens MTw data at true 60 Hz, with no D-BRAN inference.

For building a fine-tuning dataset we only need the calibrated IMU stream,
not a live pose prediction. run_dbran_live.py spends ~15-16 ms per frame on
the D-BRAN forward pass alone -- right at (or over) the 16.67 ms budget for
60 Hz, which is why long capture sessions were only sustaining ~50 Hz
effective throughput. This script does only UDP receive + calibration
(~1 ms per frame, measured), so it should comfortably sustain a real,
consistent 60 Hz -- matching Motive's hardware-synced camera capture, which
makes the later Motive<->Xsens temporal alignment far more trustworthy (a
single constant offset instead of an estimated, possibly-nonlinear rate
correction).

Usage:
    python .\\scripts\\xsens\\capture_calibrated_only.py `
      --calibration .\\configs\\xsens_calibration.json `
      --countdown 8 `
      --max_frames 10800 `
      --save_pt .\\data\\dataset_raw\\dbran_optitrack\\captura_001.pt
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
# END D-BRAN PROJECT BOOTSTRAP

import argparse
import socket
import time
from pathlib import Path
from typing import Dict, List

import torch

from dbran.xsens.calibration import XsensCalibration
from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensTorchFrame, XsensUdpReceiver
from main_path import PROJECT_ROOT


def _receive_one_frame(receiver: XsensUdpReceiver) -> XsensTorchFrame:
    while True:
        try:
            return receiver.receive(device="cpu")
        except socket.timeout:
            print("[waiting] No Xsens UDP frame received. Is the bridge streaming?")


def _drain_for_one_second(receiver: XsensUdpReceiver) -> int:
    deadline = time.perf_counter() + 1.0
    drained = 0
    while time.perf_counter() < deadline:
        try:
            receiver.receive(device="cpu")
            drained += 1
        except socket.timeout:
            pass
    return drained


def _countdown_while_draining(receiver: XsensUdpReceiver, seconds: int) -> None:
    if seconds <= 0:
        return
    for remaining in range(seconds, 0, -1):
        print(
            f"\rMove into a neutral T-pose. Capture starts in {remaining:2d} s...",
            end="",
            flush=True,
        )
        _drain_for_one_second(receiver)
    print("\r" + " " * 88 + "\r", end="", flush=True)


def _stats(values):
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(t.mean().item()),
        "p95": float(torch.quantile(t, 0.95).item()),
        "max": float(t.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--calibration",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "xsens_calibration.json"),
    )
    parser.add_argument("--countdown", type=int, default=8)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10800,
        help="Frames to capture after the countdown; 0 runs until Ctrl+C.",
    )
    parser.add_argument("--print_every", type=int, default=300)
    parser.add_argument("--save_pt", type=str, required=True)
    args = parser.parse_args()

    calibration_path = Path(args.calibration).expanduser().resolve()
    calibration = XsensCalibration.load(calibration_path)

    print("=" * 80)
    print("D-BRAN XSENS CAPTURE (calibration only, no inference)")
    print("=" * 80)
    print(f"Calibration: {calibration_path}")
    print(f"UDP input:   {args.host}:{args.port}")
    print()

    store: Dict[str, List[object]] = {
        "sequence": [],
        "host_unix_time_ns": [],
        "acc_calibrated": [],
        "ori_calibrated": [],
    }
    receive_ms: List[float] = []

    with XsensUdpReceiver(
        host=args.host, port=args.port, timeout_s=args.timeout
    ) as receiver:
        print("Waiting for the synchronized Xsens bridge...")
        initial_frame = _receive_one_frame(receiver)
        calibration.validate_device_ids(initial_frame.device_ids)

        print("Connected sensor assignment:")
        for index, role in enumerate(SENSOR_ROLES):
            print(f"  [{index}] {role:<10} -> {initial_frame.device_ids[index]}")
        print()

        input(
            "Press ENTER, then move into a neutral T-pose. "
            f"You will have {args.countdown} seconds before capture starts. "
        )
        _countdown_while_draining(receiver, args.countdown)
        receiver.sequence_gaps = 0

        print("Capture started. Perform the marked start gesture now.\n")

        frames_received = 0
        first_time = None
        while args.max_frames == 0 or frames_received < args.max_frames:
            try:
                t0 = time.perf_counter()
                frame = receiver.receive(device="cpu")
            except socket.timeout:
                print("[waiting] Xsens UDP timeout...")
                continue

            calibrated = calibration.apply_frame(
                frame, output_device="cpu", strict_device_ids=True
            )
            receive_ms.append((time.perf_counter() - t0) * 1000.0)

            store["sequence"].append(int(frame.sequence))
            store["host_unix_time_ns"].append(int(frame.host_unix_time_ns))
            store["acc_calibrated"].append(calibrated.acc)
            store["ori_calibrated"].append(calibrated.ori)

            frames_received += 1
            if first_time is None:
                first_time = frame.host_unix_time_ns

            if frames_received % args.print_every == 0:
                elapsed_s = (frame.host_unix_time_ns - first_time) / 1e9
                rate = frames_received / elapsed_s if elapsed_s > 0 else 0.0
                print(
                    f"Frame {frames_received:6d} | elapsed={elapsed_s:7.2f}s | "
                    f"effective rate={rate:6.2f} Hz | sequence_gaps={receiver.sequence_gaps}"
                )

        sequence_gaps = receiver.sequence_gaps

    total_elapsed_s = (
        (store["host_unix_time_ns"][-1] - store["host_unix_time_ns"][0]) / 1e9
        if len(store["host_unix_time_ns"]) > 1
        else 0.0
    )
    measured_rate = (
        (frames_received - 1) / total_elapsed_s if total_elapsed_s > 0 else 0.0
    )
    r_stats = _stats(receive_ms)

    print("\n" + "=" * 80)
    print("CAPTURE SUMMARY")
    print("=" * 80)
    print(f"  Frames captured:        {frames_received}")
    print(f"  Wall-clock duration:    {total_elapsed_s:.2f} s")
    print(f"  Measured effective rate:{measured_rate:7.3f} Hz (target 60 Hz)")
    print(f"  Sequence gaps:          {sequence_gaps}")
    print(
        f"  Receive+calibrate time: mean={r_stats['mean']:.3f} ms | "
        f"p95={r_stats['p95']:.3f} ms | max={r_stats['max']:.3f} ms"
    )

    destination = Path(args.save_pt).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "format": "dbran_xsens_calibrated_capture_v1",
        "sensor_roles": SENSOR_ROLES,
        "calibration_file": str(calibration_path),
        "arguments": vars(args),
        "sequence": store["sequence"],
        "host_unix_time_ns": store["host_unix_time_ns"],
        "acc_calibrated": torch.stack(store["acc_calibrated"]),
        "ori_calibrated": torch.stack(store["ori_calibrated"]),
    }
    torch.save(saved, destination)
    print(f"\nSaved calibrated capture: {destination}")

    if measured_rate < 55.0:
        print(
            "\n[warning] measured rate is still notably below 60 Hz -- "
            "something other than D-BRAN inference may be the bottleneck "
            "(UDP delivery, OS scheduling, or the native bridge itself)."
        )


if __name__ == "__main__":
    main()
