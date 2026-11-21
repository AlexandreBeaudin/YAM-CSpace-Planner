#!/usr/bin/env python3
"""Locate the foam block and X in saved views and estimate right-base positions."""

import argparse
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from planned_motion import PosePlanner, checked_vector, trajectory


LABELS = ("block_top", "x_center")
PIXEL_SCHEMA = {
    "type": "object", "properties": {
        "u": {"type": "number", "minimum": 0, "maximum": 1},
        "v": {"type": "number", "minimum": 0, "maximum": 1},
    }, "required": ["u", "v"], "additionalProperties": False,
}
JAW_LINE_SCHEMA = {
    "type": "object", "properties": dict.fromkeys(("start", "end"), PIXEL_SCHEMA),
    "required": ["start", "end"], "additionalProperties": False,
    "description": "Two separated points on the block's top face in the main image. "
                   "The line between them runs across the two opposing sides the jaws should squeeze.",
}


def load_view(path):
    raw = path.read_bytes()
    with Image.open(BytesIO(raw)) as image:
        size = image.size
        fmt = image.format
        image.verify()
    if fmt not in {"JPEG", "PNG"}:
        raise ValueError(f"{path}: expected a JPEG or PNG image")
    return size, raw, fmt.lower()


def locate(views, api_key):
    from astra_run import GatewayTransport, URL, httpx

    content = [{"type": "input_text", "text":
        "Locate the center of the upper face of the pink foam block and the "
        "center of the black X on white paper. Return image coordinates for "
        "the MAIN image only, normalized across the full width and height "
        "from the top-left pixel center (0,0) to the bottom-right pixel center "
        "(1,1). Wrist views supply context. Return null for a target if its "
        "center is occluded or ambiguous in the MAIN image. Also choose which "
        "opposing sides the jaws should squeeze and report a jaw_line using two "
        "points across the top face in the direction of jaw closure. Return "
        "null for jaw_line if the grasp direction is ambiguous."}]
    for name, (_, raw, fmt) in views.items():
        encoded = base64.b64encode(raw).decode("ascii")
        content.extend([
            {"type": "input_text", "text": f"{name.upper()} image"},
            {"type": "input_image", "image_url": f"data:image/{fmt};base64,{encoded}"},
        ])
    with httpx.Client(transport=GatewayTransport(), timeout=120) as client:
        response = client.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json={
            "model": "gpt-6-astra",
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "medium"},
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "tabletop_pixels", "strict": True,
                "schema": {"type": "object", "properties": {
                    **dict.fromkeys(LABELS, {"anyOf": [PIXEL_SCHEMA, {"type": "null"}]}),
                    "jaw_line": {"anyOf": [JAW_LINE_SCHEMA, {"type": "null"}]},
                }, "required": [*LABELS, "jaw_line"], "additionalProperties": False}}},
        })
        response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "completed":
        raise ValueError(f"Astra response status {payload.get('status')!r}")
    texts = [part["text"] for item in payload.get("output", []) if item["type"] == "message"
             for part in item["content"] if part["type"] == "output_text"]
    if len(texts) != 1:
        raise ValueError("Astra did not return a structured pixel observation")
    return json.loads(texts[0])


def calibration_from_extrinsics(source, calibration):
    """Combine the QZ main camera with station measurements in the left base."""
    if "T_leftbase_topcam" not in source["base_anchor"]:
        raise ValueError("expected the QZ left-base camera anchor")
    camera = source["cameras"]["main"]
    matrix = checked_vector(camera["K"], (3, 3))
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if not np.allclose(matrix, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], atol=1e-8):
        raise ValueError("expected a pinhole camera matrix without skew")
    if calibration.get("arm_side", "left") != "left":
        raise ValueError("the QZ calibration requires left-base station measurements")
    return {
        **calibration,
        "arm_side": "left",
        "main_size_px": camera["image_size_wh"],
        "main_intrinsics_px": [float(fx), float(fy), float(cx), float(cy)],
        "main_distortion_opencv": camera["distortion"],
        "left_base_T_main_optical": camera["base_T_cam"],
    }


def undistort_pixel(xy, coefficients):
    """Invert the five-coefficient OpenCV distortion model in normalized coordinates."""
    k1, k2, p1, p2, k3 = coefficients
    point = xy.copy()
    for _ in range(20):
        x, y = point
        r2 = x*x + y*y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        tangential = np.array([2*p1*x*y + p2*(r2 + 2*x*x),
                               p1*(r2 + 2*y*y) + 2*p2*x*y])
        if not np.isfinite(radial) or radial <= 0:
            break
        updated = (xy - tangential) / radial
        if np.linalg.norm(updated - point) <= 1e-10:
            return updated
        point = updated
    raise ValueError("pixel distortion inversion did not converge")


def project_targets(calibration, pixels, size, *, jaw_line=None):
    side = calibration.get("arm_side", "right")
    if side not in {"left", "right"}:
        raise ValueError("arm_side must be left or right")
    base = f"{side}_base"
    if tuple(calibration["main_size_px"]) != size:
        raise ValueError("main image dimensions differ from the calibration")
    transform_key = f"{base}_T_main_optical"
    transform = checked_vector(calibration[transform_key], (4, 4))
    rotation, origin = transform[:3, :3], transform[:3, 3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError(f"{transform_key} must be a rigid transform")
    fx, fy, cx, cy = checked_vector(calibration["main_intrinsics_px"], (4,))
    if fx <= 0 or fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    distortion = checked_vector(calibration.get("main_distortion_opencv", [0.]*5), (5,))
    normal = checked_vector(calibration[f"table_normal_{base}"], (3,))
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-5):
        raise ValueError(f"table_normal_{base} must be a unit vector pointing above the table")
    table_point = checked_vector(calibration[f"table_point_{base}_m"], (3,))
    block_height, hover = checked_vector(
        [calibration["block_height_m"], calibration["hover_clearance_m"]], (2,))
    if min(block_height, hover) <= 0 or normal @ (origin - table_point) <= block_height:
        raise ValueError("invalid table side, block height or hover clearance")
    if not pixels or not set(pixels) <= set(LABELS):
        raise ValueError("expected a block_top or x_center pixel")

    def project(hit, height, label):
        if hit is None:
            raise ValueError(f"{label} is unavailable in the main image")
        if not isinstance(hit, dict) or set(hit) != {"u", "v"}:
            raise ValueError(f"{label} must contain normalized u and v")
        if any(type(hit[key]) not in (int, float) for key in ("u", "v")):
            raise ValueError(f"{label} coordinates must be numeric")
        u, v = checked_vector([hit["u"], hit["v"]], (2,))
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError(f"{label} pixel falls outside the image")
        x, y = u * (size[0] - 1), v * (size[1] - 1)
        direction = undistort_pixel(np.array([(x - cx) / fx, (y - cy) / fy]), distortion)
        ray = rotation @ [*direction, 1]
        projection = normal @ ray
        if projection >= -1e-6:
            raise ValueError(f"{label} ray does not point toward the table")
        distance = (normal @ (table_point - origin) + height) / projection
        return origin + distance * ray

    targets = {"frame": base}
    for label, hit in pixels.items():
        if label == "block_top":
            surface = project(hit, block_height, label)
            targets["block_top_m"] = surface.tolist()
            targets["above_block_m"] = (surface + hover * normal).tolist()
        else:
            surface = project(hit, 0., label)
            targets["x_surface_m"] = surface.tolist()
            targets["above_x_m"] = (surface + (block_height + hover) * normal).tolist()
    if jaw_line is not None:
        if "block_top" not in pixels or not isinstance(jaw_line, dict) or set(jaw_line) != {"start", "end"}:
            raise ValueError("jaw_line requires block_top and two endpoints")
        start = project(jaw_line["start"], block_height, "jaw_line.start")
        end = project(jaw_line["end"], block_height, "jaw_line.end")
        direction = end - start
        if np.linalg.norm(direction) < .005:
            raise ValueError("jaw_line endpoints are too close to determine an angle")
        targets[f"jaw_direction_{base}"] = (direction / np.linalg.norm(direction)).tolist()
    return targets


def estimate_targets(calibration, pixels, size):
    if not isinstance(pixels, dict) or set(pixels) - {"jaw_line"} != set(LABELS):
        raise ValueError("pixel observation must contain block_top and x_center")
    return project_targets(calibration, {label: pixels[label] for label in LABELS}, size,
                           jaw_line=pixels.get("jaw_line"))


def top_down_grasp_rotation(direction, normal, current_rotation):
    # The linear-4310 grasp_site +Y is the slide axis and +Z points out of the jaws.
    z_axis = -np.asarray(normal, dtype=float)
    y_axis = np.array(direction, dtype=float, copy=True)
    y_axis -= z_axis * (z_axis @ y_axis)
    lateral_length = np.linalg.norm(y_axis)
    if lateral_length < 1e-6:
        raise ValueError("jaw direction must lie along the table")
    y_axis /= lateral_length
    # Swapping the opposing jaws gives the same grasp line. Use the nearer twist.
    x_axis = np.cross(y_axis, z_axis)
    if x_axis @ current_rotation[:, 0] + y_axis @ current_rotation[:, 1] < 0:
        x_axis = -x_axis
        y_axis = -y_axis
    return np.column_stack((x_axis, y_axis, z_axis))


def grasp_settings(calibration, setup):
    depth, release_gap = checked_vector(
        [setup["grasp_depth_below_top_m"], setup["release_clearance_m"]], (2,))
    block_height = float(calibration["block_height_m"])
    hover_gap = float(calibration["hover_clearance_m"])
    normal = np.asarray(calibration[f'table_normal_{calibration.get("arm_side", "right")}_base'])
    if not (0 < depth < block_height and 0 <= release_gap < hover_gap):
        raise ValueError("grasp depth or release clearance is outside the modeled block/hover range")
    if setup["arm_type"] not in {"yam", "yam_pro"}:
        raise ValueError("arm_type must be yam or yam_pro")
    return depth * normal, (block_height - depth + release_gap) * normal


def action_segments(action, targets, settings, *, open_before_grasp=False):
    depth_offset, release_offset = settings
    if action == "move_above_block":
        return [("approach", targets["above_block_m"], None)]
    if action == "grasp":
        grasp = np.asarray(targets["block_top_m"]) - depth_offset
        segments = [("lower", grasp, None), ("close", None, 0.), ("lift", targets["above_block_m"], None)]
        return [("open", None, 1.)] + segments if open_before_grasp else segments
    if action == "move_above_x":
        return [("transfer", targets["above_x_m"], None)]
    put_down = np.asarray(targets["x_surface_m"]) + release_offset
    return [("lower", put_down, None), ("open", None, 1.), ("retract", targets["above_x_m"], None)]


def plan_action(planner, action, segments, rotation, start, low, high):
    rows = []
    current = start.copy()
    for segment, position, opening in segments:
        try:
            if position is None:
                goal = current.copy()
                goal[6] = opening
            else:
                goal = planner.solve(current, np.asarray(position), rotation, low, high)
            points = trajectory(current, goal)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"{action}/{segment} failed to plan: {exc}") from exc
        rows.append((segment, points))
        current = goal
    return rows


def preview_trajectory(targets, calibration, setup):
    """Plan the four actions in joint space for offline review."""
    start = checked_vector(setup["start_joints"], (7,))
    low, high = checked_vector(setup["joint_limits"], (7, 2)).T
    if (np.any(low >= high) or np.any(start < low) or np.any(start > high)
            or low[6] != 0 or high[6] != 1 or start[6] < .95):
        raise ValueError("expected a start within station limits with the gripper open")
    settings = grasp_settings(calibration, setup)
    planner = PosePlanner(arm_type=setup["arm_type"])
    base = targets["frame"]
    if f"jaw_direction_{base}" in targets:
        rotation = top_down_grasp_rotation(targets[f"jaw_direction_{base}"],
                                           calibration[f"table_normal_{base}"], planner.fk(start)[1])
    else:
        if f"{base}_grasp_rotation" not in setup:
            raise ValueError(f"IK preview requires a jaw_line or a saved {base}_grasp_rotation")
        rotation = checked_vector(setup[f"{base}_grasp_rotation"], (3, 3))
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError(f"expected a proper {base}_grasp_rotation")
    phases = []
    current = start.copy()
    for action in ("move_above_block", "grasp", "move_above_x", "place"):
        rows = plan_action(planner, action, action_segments(action, targets, settings),
                           rotation, current, low, high)
        current = rows[-1][1][-1]
        phases.append({"action": action, "segments": [
            {"segment": segment, "waypoints": points.tolist()} for segment, points in rows]})
    return {
        "arm_type": setup["arm_type"],
        "gripper_type": "linear_4310",
        "collision_checked": False,
        "control_hz": 30,
        "start_joints": start.tolist(),
        f"target_rotation_{base}": rotation.tolist(),
        "phases": phases,
    }


def main():
    from astra_run import httpx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True, help="Main frame matching the calibration")
    parser.add_argument("--right", type=Path, help="Right wrist frame for visual context")
    parser.add_argument("--left", type=Path, help="Optional left wrist frame for visual context")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--base-extrinsics", type=Path, help="QZ camera JSON for left-base station measurements")
    parser.add_argument("--ik-setup", type=Path, help="Include a full offline IK trajectory preview")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--astra", action="store_true", help="Send supplied images to RHOS")
    source.add_argument("--pixels", type=Path, help="Reuse a saved pixel observation")
    args = parser.parse_args()
    if args.astra and not os.environ.get("RHOS_API_KEY"):
        parser.error("--astra needs RHOS_API_KEY")
    try:
        views = {name: load_view(path) for name, path in
                 (("main", args.main), ("right", args.right), ("left", args.left)) if path}
        calibration = json.loads(args.calibration.read_text())
        if args.base_extrinsics:
            calibration = calibration_from_extrinsics(json.loads(args.base_extrinsics.read_text()), calibration)
        if tuple(calibration["main_size_px"]) != views["main"][0]:
            raise ValueError("main image dimensions differ from the calibration")
        pixels = (locate(views, os.environ["RHOS_API_KEY"]) if args.astra
                  else json.loads(args.pixels.read_text()))
        targets = estimate_targets(calibration, pixels, views["main"][0])
        report = {
            "source": "gpt-6-astra" if args.astra else "saved_pixels",
            "image_sha256": {name: hashlib.sha256(view[1]).hexdigest() for name, view in views.items()},
            "pixels": pixels,
            "estimates": targets,
        }
        if args.ik_setup:
            report["trajectory_preview"] = preview_trajectory(
                targets, calibration, json.loads(args.ik_setup.read_text()))
    except (ImportError, KeyError, OSError, TypeError, ValueError, httpx.HTTPError) as exc:
        parser.exit(1, f"Target estimation failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
