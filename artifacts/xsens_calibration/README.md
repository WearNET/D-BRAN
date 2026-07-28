# Validated Xsens Calibration Artifacts

This directory stores the calibration artifacts used for the first validated
D-BRAN Xsens integration.

Recommended tracked files:

```text
xsens_calibration_capture_v2.pt
xsens_calibration_test_v2.pt
```

The active calibration remains at:

```text
configs/xsens_calibration.json
```

These files are hardware- and placement-specific. They contain the assigned
MTw device IDs and a static T-pose calibration for the current six-sensor
setup. A new calibration should be created whenever sensor placement changes.
