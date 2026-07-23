# Native Xsens bridge

This directory contains the Windows C++ bridge that acquires six synchronized
Xsens MTw2 samples through the Xsens Device API and transmits one fixed-size UDP
datagram per synchronized frame to the Python D-BRAN process.

## Sensor order

1. LeftArm
2. RightArm
3. LeftLeg
4. RightLeg
5. Head
6. Hip/root

## Default stream

- Destination: `127.0.0.1:9763`
- Rate: `60 Hz`
- Datagram size: `920 bytes`
- Protocol version: `1`

The C++ bridge performs acquisition and synchronization only. The 26-frame
online window remains inside `dbran.pipeline.DBranPipeline`.

## Build

The versioned files in this package should be copied to their active names:

```text
src/xsens_stream_bridge/xsens_stream_bridge.cpp
build.ps1
run.ps1
```

Then run:

```powershell
.\build.ps1 xsens_stream_bridge
.\run.ps1 xsens_stream_bridge
```

Start the Python receiver before opening the capture gate so the first streamed
frame can be observed.
