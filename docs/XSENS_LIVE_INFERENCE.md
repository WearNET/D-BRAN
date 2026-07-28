# Xsens Live D-BRAN Inference

This stage connects the validated Xsens calibration directly to the reusable
D-BRAN online pipeline.

```text
Six Xsens MTw sensors
        ↓
Native synchronized UDP bridge
        ↓
XsensCalibration.apply_frame()
        ↓
DBranPipeline.forward_online()
        ↓
pose [24, 3, 3] + root translation [3]
```

Unity is intentionally not part of this test. The objective is to validate
stream integrity, calibrated model input, online inference, output rotations,
translation, and the 60 Hz processing budget before introducing another
software process.

## Prerequisites

- The native Xsens bridge must compile and stream at 60 Hz.
- `configs/xsens_calibration.json` must exist.
- The six connected MTw IDs must match the calibration.
- `dbran/pipeline.py` must have passed the offline and online equivalence test.
- PyTorch must detect the intended CUDA device.

## First live test

Start the native bridge in one PowerShell terminal:

```powershell
cd C:\Users\kevin\D-BRAN\native\xsens_bridge
.\run.ps1 xsens_stream_bridge
```

Wait for all six MTw sensors and begin measurement.

In another PowerShell terminal:

```powershell
conda activate TransPose
cd C:\Users\kevin\D-BRAN

python .\scripts\xsens\run_dbran_live.py `
  --calibration .\configs\xsens_calibration.json `
  --device cuda `
  --countdown 8 `
  --max_frames 600 `
  --print_every 60 `
  --save_pt .\data\logs\dbran_live_test_v1.pt
```

After pressing ENTER, use the eight-second countdown to adopt the neutral
T-pose. Hold it until the script reports that the 26-frame temporal window is
ready. Then perform slow, controlled movements.

The script checks:

- 60 Hz stream rate;
- synchronized packet counters;
- UDP sequence gaps;
- calibration time;
- D-BRAN inference time;
- total processing time;
- end-to-end host frame age;
- output finiteness;
- local-pose rotation validity;
- accumulated root translation.

## Temporal behavior

The online model uses:

```text
20 past frames + current frame + 5 future frames = 26 frames
```

At 60 Hz, the five future frames introduce an algorithmic delay of about
83.3 ms. The script separately reports processing time and host-frame age.

The first outputs are generated from a partially initialized window. The
script reports a valid full-window output only after all 26 positions have
been filled.

## CUDA streams

The first live validation should run without `--cuda_streams`, matching the
pipeline implementation already validated against the profiler.

CUDA streams can be tested later with:

```powershell
python .\scripts\xsens\run_dbran_live.py `
  --device cuda `
  --cuda_streams `
  --max_frames 600
```

Treat it as a separate performance experiment and compare its output against
the validated non-streamed path.

## Saving results

When `--save_pt` is provided, the output contains:

- raw local Xsens acceleration;
- raw sensor-to-global Xsens orientation;
- calibrated acceleration and orientation;
- D-BRAN local pose matrices;
- root translation;
- temporal-window flags;
- sequence and packet metadata;
- calibration, inference, processing, and host-age measurements.

Do not use an unlimited run together with `--save_pt` unless sufficient memory
is available.
