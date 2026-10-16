---
name: x2real-gendata
description: Discover, validate, compose, launch, and inspect X2Real manual gendata sessions from the current checkout. Use for demonstration collection with a keyboard, Vuer/WebXR controllers, or WebSocket master arms, including device setup and HDF5 output diagnosis. Excludes automatic planner collection, replay, training, and policy evaluation.
---

# X2Real Manual Gendata

Use `manaenv/scripts/x2real_collect.py` as the collection entry point. Here,
gendata means human-operated demonstration collection on public ID scenes.
Pass `--mode gendata` explicitly; omitting it retains the same collection default.
The collector needs no Policy Space service.

## Resolve the session from live sources

Run commands from the repository root. Read the collector's `--help`, its
`_METHODS` and `_ROBOTS` declarations, and the selected configuration under
`manaenv/defination/datacollection/`. Use the selected simulation interpreter;
when it has another editable ManaEnv installed, pin `PYTHONPATH` to this checkout.

Use the [README collection instructions](../../../README.md#manual-collection) or
[Chinese instructions](../../../README.zh-CN.md#manual-collection-zh) for keyboard
controls. Device setup is described below. Do not copy device addresses,
private configuration, asset locations, or environment names from another checkout.

Preserve the user's method, robot/board, case, gripper, demonstration target,
episode timeout, seed, output directory, and visualization choices. Discover
eligible cases with the collector rather than the broader evaluation catalog:

```bash
<python> manaenv/scripts/x2real_collect.py --mode gendata --list
<python> manaenv/scripts/x2real_collect.py --mode gendata --list --board <board>
```

The CLI accepts one exact `--env`, or filters with `--board`, `--task`, and
`--gripper` that resolve to exactly one case. Do not silently choose a robot,
task, or input device when the request leaves those decisions open. Inspect the
live choices, then ask only for the missing selection needed to compose the run.
Collection uses one environment and enables cameras internally; evaluation
parallelism and episode-budget flags do not apply.

## Resolve the device and preflight

- `--teleop keyboard`: requires the Kit window, `--viz kit`.
- `--teleop vuer`: browser controller input; `--teleop vr` is its alias.
  Check the selected interpreter for the optional `teleop-vr` dependency.
  If installation is requested, use the same environment and the documented
  `python -m pip install -e '.[teleop-vr]'` command. Resolve `--vuer-host` and
  `--vuer-port` from the requested headset topology; loopback is for local use.
  A real collection starts the Vuer input server; `--dry-run` does not.
- `--teleop masterslave`: connects as a client to an existing master-arm server.
  Resolve `--master-address` and `--master-port` from the request or a verified
  device deployment. A loopback default is not proof of a connected master.
  Use `MANAENV_MASTER_TOKEN` when authentication is required; do not print or
  embed its value in commands or YAML. Do not start or replace the remote server
  as a side effect of composing a collection command.

For simulator installation or repair, use the repository's `$x2real-install`
workflow. Retain the current asset-root setting or the repository asset default;
do not substitute internal asset servers.

For Vuer, open the page printed by the input server in a WebXR-capable browser
with the required secure context and certificate setup. Left `X` starts/pauses
following, left `Y` saves, and right `A` discards and resets. Triggers control
grippers. Supported mobile robots also consume base/lift inputs.

The master server sends JSON with `left` and `right` objects. Each `joint_pos`
contains six arm joints followed by a gripper position:

```json
{"left": {"joint_pos": [0, 0, 0, 0, 0, 0, 0]},
 "right": {"joint_pos": [0, 0, 0, 0, 0, 0, 0]}}
```

Set `joint_signs` and `joint_offsets` in the selected planner configuration to
match the device. In the Kit window, `Enter` saves and `R` discards and resets.
Disconnected input or input older than one second pauses physics.

```bash
<python> manaenv/scripts/x2real_collect.py --mode gendata --check
<python> manaenv/scripts/x2real_collect.py --mode gendata --asset-check
```

Run the configuration check before a launch. Run the asset check when preparing
an actual session or diagnosing uncertain assets. Missing assets or device
prerequisites block that session, not static selection or command preparation.
Report the missing prerequisite without claiming that a dry-run tested it.

## Compose and preview the requested command

```bash
<python> manaenv/scripts/x2real_collect.py \
  --mode gendata \
  --env <exact-env> \
  --teleop <keyboard|vuer|masterslave> \
  --num-demos <successful-demonstrations> \
  --save-path <local-output-directory> \
  --seed <integer-or-random> \
  --viz <kit|none> \
  --dry-run
```

Add the selected device's endpoint arguments. Add `--episode-length-s` when the
user requests a different timeout; otherwise retain the case timeout. `none`
selects headless execution through IsaacLab's visualizer selection and is
unavailable for keyboard input.

Confirm the preview's exact environment, input method, demonstration target,
resolved seed, output directory, and generated launch arguments. This CLI does
not accept arbitrary simulator passthrough flags. Remove `--dry-run` only when
execution is requested or already authorized; a command-only request ends with
the validated command. Do not overwrite existing recordings to restart a session.

The collector forwards `--mode gendata` to `manaenv/scripts/main.py`. At that
lower-level entry, gendata selects manual-input planner orders and benchmark
selects policy orders. An explicit mode with no matching orders is an error;
it must not silently execute the other kind of rollout.

## Observe recording and verify artifacts

For a requested run, verify Kit startup, the selected scene and cameras, input
readiness, finite actions with the robot's declared layout, and an actual save.
Kit collection opens previews of the existing sensor render products. Supported
G-jaw wrist views show display-only tip crosshairs; H-jaw skips the uncalibrated
marker. `MANAENV_EX001_6R_TIP_CROSSHAIR=0` disables markers without disabling
previews or recording. Kit also shows the dev-6.0-style `Sim FPS` rate and
per-step time. `--viz none` creates neither camera nor FPS windows or markers.
VR pause and disconnected/stale master input pause physics; they do not establish
successful collection. Diagnose a wait at the device boundary before changing
the task or recording settings.

Use the JSON report's `datasets` paths to inspect every saved HDF5 file:

```bash
<python> manaenv/scripts/x2real_collect.py --mode gendata --inspect <saved-hdf5>
```

`--num-demos` counts successful persisted demonstrations, not attempts or process
completion. Sum episodes whose stored `success` is true and verify nonzero
recorded frames. Scene rebuilding may create several HDF5 files, so a file count
is not a demonstration count. Check language/action metadata when diagnosing
incorrect labels or action layouts. Discarded, failed, short, and interrupted
attempts must not be counted as successful demonstrations; preserve saved data.

Report the exact command, interpreter, selected case and device, output/report
paths, and achieved demonstration count. Distinguish configuration/dry-run checks
from simulator, camera, hardware-input, and HDF5 export validation. Complete a
requested execution only when its saved target is reached; otherwise state the
observed stopping point and remaining device or runtime requirement.
