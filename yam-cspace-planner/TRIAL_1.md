# Foam-block pick-and-place trial 1

## Relative end-effector trial

For the first full pick-and-place using visual feedback, use
`tabletop-relative`. Astra gets the overhead and wrist images and the
grasp-point pose from joint feedback. It chooses relative left-arm
translations, wrist rotations, and gripper actions from those observations.
IK computes the joint targets. Each action is followed by a new observation.
For a clear transit, one call can request up to 20 cm of translation and
1 rad of rotation. Near the block or table, the limits are 5 cm and 0.25 rad.
Both phases use the existing 0.80 rad per-joint limit.
The task uses the arm model from the station profile and needs no camera
extrinsics, tabletop measurements, or calibration files.

Run the Bash setup block in [section 1](#1-check-the-station-files). Start the
safety daemon and camera preview as in [section 2](#2-start-the-safety-daemon-in-terminal-a)
and [section 3](#3-start-the-camera-preview-in-terminal-b). Keep the left
gripper open and the workspace clear. Use the station’s normal attended
operation. This mode enforces motion and tracking limits. Clearance is judged
from the views and by the person at the station.

```bash
"$YAM_PY" "$RUNNER" --task tabletop-relative --attempt 1 --check-station
read -r -s -p 'RHOS API key: ' RHOS_API_KEY
printf '\n'
export RHOS_API_KEY
"$YAM_PY" "$RUNNER" --task tabletop-relative --attempt 1 \
  --run --left-channel-confirmed can_yam_l
```

The run appears in `yam-cspace-planner/runs/relative-001/`. Open its
`replay.html`. To export MP4, follow [section 5](#5-open-the-replay-and-export-mp4)
and set `RUN_DIR="$PROJECT/runs/relative-001"`.

## Absolute Cartesian trial

The Cartesian runner is wired into Inspect Robots. It sends new overhead and
wrist images and joint readback after each action. The policy locates the
targets and the direction the jaws should close in the current overhead view,
then calls one of four tools. Each tool
projects its target into the left arm’s base frame and builds an IK trajectory.
Four movement actions plus a final observation and `done` are the best-case run.
The runner records corrections and retry calls in the same episode.

## 1. Check the station files

Use Bash for the commands below. Run this block in each terminal.

```bash
cd ~/workspace/jiaxin/shurui/yam-astra-experiments
YAM_PY=/home/rhos/workspace/YAM/gello_software/.venv/bin/python
YAM_ROOT=/home/rhos/workspace/YAM
PROJECT="$PWD/../yam-cspace-planner"
RUNNER=integrations/inspect_robots/astra_run.py
YAM_SETUP=/home/rhos/.config/yam/setup/version-a0a265371ebd9dbd
export PYTHONDONTWRITEBYTECODE=1
```

Run the offline preflight first.

```bash
"$YAM_PY" "$RUNNER" --task tabletop-ik --attempt 1
```

Expect `can_yam_r` on the right and `can_yam_l` on the left.

The QZ camera file anchors the overhead camera to the left arm’s base. The
commands below therefore select the left arm and keep the right arm parked.
Verify the overhead projection against this station’s actual main stream and
use the table plane measured in the left base.

| File | Contents |
| --- | --- |
| `base_extrinsics_final.json` | Haoxiang’s QZ camera file. The runner reads its main camera intrinsics, distortion coefficients, and camera-to-left-base transform. |
| `tabletop-calibration.local.json` | Supply `table_point_left_base_m`, `table_normal_left_base`, `block_height_m`, and `hover_clearance_m`. Set `main_preview_matches_calibration` to true after verifying the actual preview projection. |
| `tabletop-ik.local.json` | Verified arm variant, grasp depth, and release clearance. Astra chooses the jaw angle from the image. The live run takes start joints and limits from the station. |
| `tabletop_clearance.py` | The station’s validated `check_path(path, context, observation)` implementation. It must verify the swept motion of each segment, accounting for the table, fixtures, parked arm, gripper, block, intended grasp/release contact, and tracking uncertainty. The path is an array of the start and all proposed left-arm waypoints. The context includes segment boundaries, measured start, parked-arm joints, and the current target. The checker returns `True` only for a verified path. |

The calibration key names and an offline IK preview command are in the [target
estimation guide](../yam-astra-experiments/integrations/inspect_robots/TABLETOP_IK.md).
Submit the four files to offline preflight before acquiring the arm.

```bash
TRIAL_ARGS=(
  --task tabletop-ik --attempt 1
  --tabletop-extrinsics "$PROJECT/base_extrinsics_final.json"
  --tabletop-calibration "$PROJECT/tabletop-calibration.local.json"
  --tabletop-setup "$PROJECT/tabletop-ik.local.json"
  --tabletop-checker "$PROJECT/tabletop_clearance.py"
)
"$YAM_PY" "$RUNNER" "${TRIAL_ARGS[@]}"
```

Expect both `tabletop_configuration_loaded` and `tabletop_checker_loaded` to
read `true`, with `active_arm` set to `left`. This verifies loading. Validate the checker on this station before use.

## 2. Start the safety daemon in terminal A

If the station daemon is already running, use it and continue to step 3.
Otherwise use the lab’s supervised startup procedure. The command below
initializes the followers. Ensure the arms are physically supported as required
by the lab, the stop control is reachable, and the daemon is the sole CAN owner.

```bash
PYTHONPATH="$YAM_ROOT/gello_software:$YAM_ROOT/i2rt" \
  "$YAM_PY" -u -m experiments.yam_safety_daemon \
  --left-config-path "$YAM_SETUP/left.yaml" \
  --right-config-path "$YAM_SETUP/right.yaml" \
  --hardware-port 6001 --safety-port 6002
```

Leave it running in confirmed `SAFE_HOLD`.

## 3. Start the camera preview in terminal B

If the station preview is already running, open it. Otherwise start it from
the repository root with the local station profile.

```bash
"$YAM_PY" -u preview_cameras.py
```

Open [camera preview](http://127.0.0.1:8765/). Confirm main, right, and left are
live and correctly identified. Place the foam block and X where the main view
can see them. Start with the left gripper open.

## 4. Check station and run from terminal C

Use the `TRIAL_ARGS` block above in this terminal.

```bash
"$YAM_PY" "$RUNNER" "${TRIAL_ARGS[@]}" --check-station
```

Wait for `Station observations and daemon checks passed.` This checks the
running daemon, its code and motor-config identity, measured feedback, and
camera observations without starting an episode.

```bash
read -r -s -p 'RHOS API key: ' RHOS_API_KEY
printf '\n'
export RHOS_API_KEY
"$YAM_PY" "$RUNNER" "${TRIAL_ARGS[@]}" --run --left-channel-confirmed can_yam_l
```

The last command starts an attended motion session. If you stop the run with
Ctrl+C, keep the daemon running and check the final hold. Reuse the recordings
from the interrupted attempt and increment `--attempt` for a fresh run.

## 5. Open the replay and export MP4

Trial 1 writes to `yam-cspace-planner/runs/run-001/`. Open `replay.html` for the
same synchronized main and left-wrist video panels and Model, Tools, Motion,
and Images timeline as the earlier project. See `attempt-result.json` for the
completed action list, model attempts and final hold, and `station-events.jsonl`
for target pixels, the jaw line, robot-frame target poses, planned waypoints, clearance decisions,
and measured readback. The `wire/` and `transcripts/` folders retain model and
tool evidence. Trial files contain lab images and experiment data.

```bash
RUN_DIR="$PROJECT/runs/run-001"
EXPORTER=integrations/inspect_robots/export_replay_mp4.py

"$YAM_PY" "$EXPORTER" --run-dir "$RUN_DIR" \
  --title "Foam block Cartesian pick-and-place" --html-only

PYTHONPATH="$PWD/.venv/replay-packages" \
  PLAYWRIGHT_BROWSERS_PATH="$PWD/.venv/replay-browsers" \
  "$YAM_PY" "$EXPORTER" --run-dir "$RUN_DIR" \
  --title "Foam block Cartesian pick-and-place" --speed 4 --fps 30
```

The second command creates `replay_4x.mp4` and a JSON export report. Use
`--speed 1` for real-time playback. Both exports can rebuild from the evidence
of an interrupted run.
