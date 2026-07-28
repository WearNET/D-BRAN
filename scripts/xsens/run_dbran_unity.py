"""Stream live D-BRAN output to the original TransPose Unity example.

The original TransPose Unity package expects a TCP client/server stream on
127.0.0.1:8888. Each frame is encoded as UTF-8 text using this format:

    72 local SMPL axis-angle values # 3 root-translation values $

This script keeps the validated live chain:

    six Xsens MTw sensors
        -> synchronized native UDP bridge
        -> saved Xsens calibration
        -> DBranPipeline.forward_online()
        -> original TransPose Unity TCP protocol

Start the native Xsens bridge first, then this script, and finally press Play
in the TransPose Unity Example scene when the script waits for Unity. The
stream runs continuously until the user presses Ctrl+C.
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
from collections import deque
import time
from pathlib import Path
from typing import Sequence

import torch
import articulate.math as M

from dbran.pipeline import DBranPipeline
from dbran.xsens.calibration import XsensCalibration
from dbran.xsens.protocol import SENSOR_ROLES
from dbran.xsens.receiver import XsensTorchFrame, XsensUdpReceiver
from main_path import PROJECT_ROOT

JOINT_COUNT = 24
POSE_VALUE_COUNT = JOINT_COUNT * 3
TRANSLATION_VALUE_COUNT = 3


def resolve_device(value: str) -> torch.device:
    """Resolve a requested PyTorch device and validate CUDA availability."""
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is unavailable.")
    return device


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA only when timing GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stats(values: Sequence[float]) -> dict[str, float]:
    """Return basic timing statistics."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


def print_stats(label: str, values: Sequence[float]) -> None:
    """Print one formatted timing-statistics row."""
    result = stats(values)
    print(
        f"  {label:<30} mean={result['mean']:8.3f} ms | "
        f"p50={result['p50']:8.3f} | "
        f"p95={result['p95']:8.3f} | "
        f"max={result['max']:8.3f}"
    )


def receive_one(receiver: XsensUdpReceiver) -> XsensTorchFrame:
    """Wait until one synchronized Xsens frame is available."""
    while True:
        try:
            return receiver.receive(device="cpu")
        except socket.timeout:
            print("[waiting] No Xsens UDP frame. Is the native bridge streaming?")


def drain_one_second(receiver: XsensUdpReceiver) -> None:
    """Drain incoming frames for approximately one second."""
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        try:
            receiver.receive(device="cpu")
        except socket.timeout:
            pass


def countdown(receiver: XsensUdpReceiver, seconds: int) -> None:
    """Drain live frames while the user adopts the neutral T-pose."""
    for remaining in range(seconds, 0, -1):
        print(
            f"\rMove into the neutral T-pose. Streaming starts in "
            f"{remaining:2d} s...",
            end="",
            flush=True,
        )
        drain_one_second(receiver)
    print("\r" + " " * 88 + "\r", end="", flush=True)


def warm_up(pipeline: DBranPipeline, frames: int) -> None:
    """Warm up the online pipeline without retaining temporal state."""
    if frames <= 0:
        pipeline.reset_online_state()
        return

    print(f"Warming up D-BRAN with {frames} synthetic frames...")
    acc = torch.zeros(6, 3)
    ori = torch.eye(3).repeat(6, 1, 1)

    with torch.inference_mode():
        for _ in range(frames):
            pipeline.forward_online(acc, ori)

    synchronize(pipeline.device)
    pipeline.reset_online_state()
    print("D-BRAN warm-up complete.\n")


def validate_frame(
    frame: XsensTorchFrame,
    calibration: XsensCalibration,
) -> None:
    """Validate sensor assignment and stream frequency."""
    calibration.validate_device_ids(frame.device_ids)

    if frame.update_rate_hz != calibration.update_rate_hz:
        raise RuntimeError(
            "The live Xsens stream rate does not match the calibration rate."
        )

    if frame.update_rate_hz != 60:
        raise RuntimeError(
            f"Expected a 60 Hz Xsens stream, received {frame.update_rate_hz} Hz."
        )

    print("Connected sensor assignment:")
    for index, role in enumerate(SENSOR_ROLES):
        print(f"  [{index}] {role:<10} -> {frame.device_ids[index]}")
    print()


def validate_first_output(
    pose: torch.Tensor,
    translation: torch.Tensor,
) -> None:
    """Validate the first full-window D-BRAN output."""
    if tuple(pose.shape) != (JOINT_COUNT, 3, 3):
        raise RuntimeError(
            f"Expected pose ({JOINT_COUNT}, 3, 3), got {tuple(pose.shape)}."
        )

    if translation.numel() != TRANSLATION_VALUE_COUNT:
        raise RuntimeError(
            f"Expected 3 translation values, got shape {tuple(translation.shape)}."
        )

    if not torch.isfinite(pose).all() or not torch.isfinite(translation).all():
        raise RuntimeError("The first full-window output contains non-finite values.")

    pose_cpu = pose.detach().to(device="cpu", dtype=torch.float32)
    identity = torch.eye(3).expand_as(pose_cpu)
    orthogonality_error = float(
        (pose_cpu.transpose(-1, -2) @ pose_cpu - identity).abs().amax().item()
    )
    determinant_error = float(
        (torch.linalg.det(pose_cpu) - 1.0).abs().amax().item()
    )

    print(
        f"[validated] First output: orth_err={orthogonality_error:.3e}, "
        f"det_err={determinant_error:.3e}\n"
    )


def build_transpose_message(
    local_pose: torch.Tensor,
    translation: torch.Tensor,
) -> bytes:
    """Encode one frame using the original TransPose Unity text protocol."""
    if tuple(local_pose.shape) != (JOINT_COUNT, 3, 3):
        raise ValueError(
            f"Expected local pose ({JOINT_COUNT}, 3, 3), "
            f"got {tuple(local_pose.shape)}."
        )

    # The original TransPose Unity package expects LOCAL SMPL rotations in
    # axis-angle form. Do not convert these matrices to global rotations and
    # do not apply the coordinate-basis flip used by the custom UDP receiver.
    axis_angle = M.rotation_matrix_to_axis_angle(local_pose).reshape(-1)
    if axis_angle.numel() != POSE_VALUE_COUNT:
        raise RuntimeError(
            f"Expected {POSE_VALUE_COUNT} axis-angle values, "
            f"got {axis_angle.numel()}."
        )

    pose_values = (
        axis_angle.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    translation_values = (
        translation.reshape(-1)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )

    if translation_values.size != TRANSLATION_VALUE_COUNT:
        raise RuntimeError(
            f"Expected {TRANSLATION_VALUE_COUNT} translation values, "
            f"got {translation_values.size}."
        )

    pose_text = ",".join(f"{float(value):g}" for value in pose_values)
    translation_text = ",".join(
        f"{float(value):g}" for value in translation_values
    )

    # Exact framing used by TransPose/example_server.py:
    #   pose values + '#' + translation values + '$'
    return f"{pose_text}#{translation_text}$".encode("utf-8")


def create_unity_server(host: str, port: int) -> socket.socket:
    """Create the TCP server expected by the TransPose Unity example."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run calibrated live Xsens data through D-BRAN and stream the "
            "result to the original TransPose Unity example."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--calibration",
        default=str(PROJECT_ROOT / "configs" / "xsens_calibration.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--countdown", type=int, default=8)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option. It is ignored because this "
            "version streams continuously until Ctrl+C."
        ),
    )
    parser.add_argument("--print_every", type=int, default=120)
    parser.add_argument(
        "--stats_window",
        type=int,
        default=3600,
        help=(
            "Number of recent output frames retained for timing statistics. "
            "The default keeps about one minute at 60 Hz."
        ),
    )
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--num_past_frame", type=int, default=20)
    parser.add_argument("--num_future_frame", type=int, default=5)
    parser.add_argument("--cuda_streams", action="store_true")
    parser.add_argument("--unity_host", default="127.0.0.1")
    parser.add_argument("--unity_port", type=int, default=8888)
    args = parser.parse_args()

    if args.countdown < 0 or args.max_frames < 0:
        raise ValueError("countdown and max_frames cannot be negative.")
    if args.max_frames != 0:
        print(
            "[notice] --max_frames is ignored in this continuous version. "
            "Use Ctrl+C to stop the stream cleanly."
        )
    if args.print_every <= 0 or args.stats_window <= 0 or args.warmup_frames < 0:
        raise ValueError(
            "print_every and stats_window must be positive; "
            "warmup_frames must be non-negative."
        )
    if not 1 <= args.port <= 65535 or not 1 <= args.unity_port <= 65535:
        raise ValueError("TCP and UDP ports must be between 1 and 65535.")

    device = resolve_device(args.device)
    calibration_path = Path(args.calibration).expanduser().resolve()
    calibration = XsensCalibration.load(calibration_path)

    print("=" * 80)
    print("D-BRAN -> ORIGINAL TRANSPOSE UNITY EXAMPLE")
    print("=" * 80)
    print(f"Calibration:       {calibration_path}")
    print(f"Device:            {device}")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(device)}")
    print(f"Xsens bridge UDP:  {args.host}:{args.port}")
    print(f"Unity TCP server:  {args.unity_host}:{args.unity_port}")
    print("Unity protocol:    72 local axis-angle values # 3 translation values $")
    print("Stream duration:   Continuous until Ctrl+C")
    print(f"Timing window:     Most recent {args.stats_window} output frames")
    print()

    pipeline = DBranPipeline(
        device=device,
        num_past_frame=args.num_past_frame,
        num_future_frame=args.num_future_frame,
        use_cuda_streams=args.cuda_streams,
        verbose=True,
    )
    print(f"[D-BRAN] Total parameters: {pipeline.parameter_counts()['total']:,}\n")
    warm_up(pipeline, args.warmup_frames)

    frame_period_ms = 1000.0 / calibration.update_rate_hz
    algorithmic_delay_ms = args.num_future_frame * frame_period_ms

    # Keep timing diagnostics bounded so continuous streaming does not grow
    # memory usage over long sessions. Counters below still cover the full run.
    processing_times: deque[float] = deque(maxlen=args.stats_window)
    host_age_times: deque[float] = deque(maxlen=args.stats_window)
    effective_age_times: deque[float] = deque(maxlen=args.stats_window)

    frames_received = 0
    messages_sent = 0
    send_failures = 0
    over_budget = 0
    receive_timeouts = 0
    sequence_gaps = 0
    first_checked = False
    first_receive_time: float | None = None
    last_receive_time: float | None = None

    unity_server: socket.socket | None = None
    unity_connection: socket.socket | None = None

    try:
        with XsensUdpReceiver(args.host, args.port, args.timeout) as receiver:
            print("Waiting for the synchronized Xsens bridge...")
            initial_frame = receive_one(receiver)
            validate_frame(initial_frame, calibration)

            unity_server = create_unity_server(args.unity_host, args.unity_port)
            print(
                "Unity server started. Open the original TransPose Example "
                "scene and press Play."
            )
            print("Waiting for Unity to connect...")
            unity_connection, unity_address = unity_server.accept()
            unity_connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Unity connected from {unity_address[0]}:{unity_address[1]}.\n")

            input(
                "Press ENTER, then move into the neutral T-pose. "
                f"You have {args.countdown} seconds. "
            )
            countdown(receiver, args.countdown)

            receiver.sequence_gaps = 0
            pipeline.reset_online_state()
            print(
                "Streaming started. Hold the T-pose until the 26-frame "
                "window is ready.\n"
            )

            with torch.inference_mode():
                while True:
                    try:
                        frame = receiver.receive(device="cpu")
                    except socket.timeout:
                        receive_timeouts += 1
                        continue

                    receive_time = time.perf_counter()
                    if first_receive_time is None:
                        first_receive_time = receive_time
                    last_receive_time = receive_time
                    processing_start = time.perf_counter()

                    calibrated = calibration.apply_frame(
                        frame,
                        output_device=device,
                        strict_device_ids=True,
                    )
                    output = pipeline.forward_online(
                        calibrated.acc,
                        calibrated.ori,
                    )
                    frames_received += 1
                    sequence_gaps = receiver.sequence_gaps

                    if not output.has_full_window:
                        continue

                    if not first_checked:
                        validate_first_output(output.pose, output.translation)
                        first_checked = True
                        print(
                            f"[ready] Full temporal window at input frame "
                            f"{output.input_frame_index}. Sending to Unity.\n"
                        )

                    message = build_transpose_message(
                        output.pose,
                        output.translation,
                    )

                    try:
                        unity_connection.sendall(message)
                        messages_sent += 1
                    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                        send_failures += 1
                        raise RuntimeError(
                            "The TransPose Unity client disconnected. Stop Play "
                            "mode, restart this script, and press Play again."
                        ) from exc

                    synchronize(device)
                    processing_ms = (
                        time.perf_counter() - processing_start
                    ) * 1000.0
                    host_age_ms = (
                        time.time_ns() - frame.host_unix_time_ns
                    ) / 1_000_000.0
                    effective_age_ms = host_age_ms + algorithmic_delay_ms

                    processing_times.append(processing_ms)
                    host_age_times.append(host_age_ms)
                    effective_age_times.append(effective_age_ms)

                    if processing_ms > frame_period_ms:
                        over_budget += 1

                    if messages_sent % args.print_every == 0:
                        elapsed = receive_time - first_receive_time
                        measured_rate = (
                            (frames_received - 1) / elapsed if elapsed > 0 else 0.0
                        )
                        root = output.translation.detach().reshape(-1).to("cpu")
                        print(
                            f"Sent {messages_sent:5d} | "
                            f"rate={measured_rate:6.2f} Hz | "
                            f"processing={processing_ms:7.3f} ms | "
                            f"effective age={effective_age_ms:7.3f} ms"
                        )
                        print(
                            f"  Root=[{float(root[0]): .4f}, "
                            f"{float(root[1]): .4f}, "
                            f"{float(root[2]): .4f}] | "
                            f"gaps={receiver.sequence_gaps}\n"
                        )

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        if unity_connection is not None:
            try:
                unity_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            unity_connection.close()

        if unity_server is not None:
            unity_server.close()

    elapsed = (
        last_receive_time - first_receive_time
        if first_receive_time is not None and last_receive_time is not None
        else 0.0
    )
    measured_rate = (
        (frames_received - 1) / elapsed
        if frames_received > 1 and elapsed > 0
        else 0.0
    )

    print("\n" + "=" * 80)
    print("TRANSPOSE UNITY STREAM SUMMARY")
    print("=" * 80)
    print(f"  Xsens frames received:       {frames_received}")
    print(f"  Unity messages sent:         {messages_sent}")
    print(f"  Unity send failures:         {send_failures}")
    print(f"  Xsens sequence gaps:         {sequence_gaps}")
    print(f"  Xsens receive timeouts:      {receive_timeouts}")
    print(f"  Measured receive rate:       {measured_rate:.3f} Hz")
    print(
        f"  Messages over {frame_period_ms:.3f} ms budget: "
        f"{over_budget}/{messages_sent}"
    )
    print()
    print(
        f"  Timing statistics window:    latest "
        f"{len(processing_times)}/{args.stats_window} output frames"
    )
    print()
    print_stats("Inference + encode + TCP send", processing_times)
    print_stats("Newest-frame host age", host_age_times)
    print_stats("Effective output age", effective_age_times)
    print("\nD-BRAN TRANSPOSE UNITY STREAM COMPLETED")


if __name__ == "__main__":
    main()
