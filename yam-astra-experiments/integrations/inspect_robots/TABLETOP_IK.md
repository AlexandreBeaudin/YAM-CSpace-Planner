# Foam-block pick-and-place on rhos-nuc-3

Use these commands from the root of the Astra experiment checkout. The local
`profiles/shurui-mcap.yaml` points to the station's existing setup and safety
daemon source. Its left and right channels are `can_yam_l` and `can_yam_r`.

## Next station session

1. Record a short left-arm diagnostic with the table clear, using the station's
   normal attended controls. Capture a small free-space pitch movement and a
   brief hold in each direction. Compare `driver_target` against `position`
   for `left_j3` and review `velocity`, `feedback_effort`, and
   `gravity_effort`. Trial 2 reached the tracking stop on the fourth move.
   The tabletop runner now saves these driver fields with the measured
   samples and identifies the joint and target in a tracking stop event.
2. Register the overhead preview to this station's left base. Use measured
   stationary grid intersections spread across the tabletop for the fit and
   reserve a separate set of intersections to verify it. The grid cells are
   42 mm left-to-right and 37 mm front-to-back. The pink cube is 60 mm per
   side. The underside of the fixed black left-arm mounting plate is 30 mm
   above the tabletop, per the station measurement.
   The imported QZ transform projected the Trial 2 block to about
   61 cm sideways and did not give a convergent hover IK solution in the
   offline check.
3. Generate and review the full hover path from the current arm pose. Check
   the swept clearance of the table, parked right arm, and gripper. Run
   `tabletop-ik --hover-only`, then check arrival in the wrist and overhead
   views and review the tracking trace. Astra picks the block center and
   jaw angle from the current images. The runner ends after the hover and
   its final observation.
4. After the hover passes, run the complete `tabletop-ik` trial. Compare
   model wait, path execution time, and pose error with Trial 2. Use the
   recorded views after each action to confirm the grasp and placement.

For the first step, start this capture in a second terminal with the daemon
running. It samples the left driver's diagnostics for 20 seconds at up to 10 Hz.
Run it from the experiment repo.

```bash
/home/rhos/workspace/YAM/gello_software/.venv/bin/python -B - <<'PY'
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "yam3-sunday")
from capture_observations import DiagnosticsClient, measured_follower

output = Path(f"left-tracking-{time.time_ns()}.jsonl")
client = DiagnosticsClient()
try:
    with output.open("x") as stream:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            started = time.monotonic()
            snapshot, received_ns, _ = client.read(0.5)
            sample = measured_follower(snapshot, "left")
            stream.write(json.dumps({"received_time_ns": received_ns, **sample}) + "\n")
            time.sleep(max(0.0, 0.1 - (time.monotonic() - started)))
finally:
    client.close()
    print(output.resolve())
PY
```

Once the station calibration, setup, and clearance checker are in place,
check the station with the following command. For the attended hover, replace
`--check-station` with `--run --left-channel-confirmed can_yam_l`. The hover has
its own output directory at `../yam-cspace-planner/runs/hover-001/`.

```bash
/home/rhos/workspace/YAM/gello_software/.venv/bin/python -B \
  integrations/inspect_robots/astra_run.py --task tabletop-ik --hover-only --attempt 1 \
  --tabletop-calibration ../yam-cspace-planner/tabletop-calibration.local.json \
  --tabletop-setup ../yam-cspace-planner/tabletop-ik.local.json \
  --tabletop-checker ../yam-cspace-planner/tabletop_clearance.py --check-station
```

```bash
YAM_PY=/home/rhos/workspace/YAM/gello_software/.venv/bin/python
"$YAM_PY" -B integrations/inspect_robots/astra_run.py --attempt 1
```

## Estimate targets and preview the full IK trajectory

Haoxiang’s `base_extrinsics_final.json` supplies a 640×480 main-camera
matrix, distortion coefficients, and a transform anchored to the left base.
Use `--base-extrinsics PATH` with saved, raw main images to apply the supplied
distortion coefficients in OpenCV order. Supply `table_point_left_base_m`
and `table_normal_left_base` in the station calibration JSON, along with
`block_height_m` and `hover_clearance_m`. The camera file supplies the image
size, intrinsics, and camera pose. For the live runner, the corresponding
argument is `--tabletop-extrinsics PATH`. Check the projection against this
station before setting `main_preview_matches_calibration`.

Save a rectified main image from the current tabletop and a contemporaneous
right wrist image. The optional left wrist image adds context. Astra returns
the center of the block's top face, the center of X, and two points across
the block in the selected jaw-closing direction in the main image.
The script intersects their camera rays with the top-of-block and table planes,
respectively. Each new scene supplies its own pixel locations.

```bash
"$YAM_PY" -B integrations/inspect_robots/tabletop_targets.py \
  --main main_rectified.png --right right.jpg --astra \
  --calibration tabletop-calibration.local.json \
  --ik-setup tabletop-ik.local.json > tabletop-preview.local.json
```

Set `RHOS_API_KEY` in the environment for `--astra`. For offline replay, use
`--pixels pixels.json` with a JSON object containing `block_top` and `x_center`,
each with `u` and `v` normalized from the first to last pixel center. The
`jaw_line` field has `start` and `end` points in the same format. With only
the calibration file and images, omit `--ik-setup` to estimate the target positions.

Use measurements from this station for the calibration JSON.

- `main_size_px` holds the calibrated image width and height.
- `main_intrinsics_px` holds `[fx, fy, cx, cy]` for the rectified image.
- `right_base_T_main_optical` is the 4×4 transform from the camera optical
  frame to the right follower base, with translation in meters.
- `table_point_right_base_m` is a point on the tabletop in the right base.
  `table_normal_right_base` is the unit vector from the table into free space.
- `block_height_m` is the foam block thickness. `hover_clearance_m` is the
  vertical clearance above the source block and above the held block's
  final resting elevation at X.

The IK setup JSON supplies the measured seven-element `start_joints`, the
seven `[low, high]` station `joint_limits`, and `arm_type` equal to `yam` or
`yam_pro` for a linear-4310 gripper. The first six joints use radians. The
last is normalized to 0–1. Supply `grasp_depth_below_top_m` and
`release_clearance_m`. The planner uses the jaw line to set the angle for a
top-down grasp and carries it through placement. To replay older pixel
observations without a jaw line, supply the saved 3×3
`right_base_grasp_rotation` in the offline IK setup.

The preview groups the interpolated path into approach, grasp, transfer, and
place. It reports `collision_checked: false`. Hardware integration still
requires a verified tool center point, whole-path clearance for the actual
station and held block, and grasp and placement feedback. The live
`tabletop-ik` task takes Astra's jaw line at the block hover and checks the
resulting trajectory before motion.
