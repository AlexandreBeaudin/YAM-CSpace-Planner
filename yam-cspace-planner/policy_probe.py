#!/usr/bin/env python3
"""Measure Astra's pick-and-place decisions on a mock tabletop."""

import argparse
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "gpt-6-astra"
URL = "https://api.rhos.cc/v1/responses"
START = (0.25, -0.20, 0.18)
ABOVE_BLOCK = (0.43, -0.10, 0.10)
ABOVE_X = (0.40, 0.17, 0.10)
WORKSPACE = ((0.15, 0.55), (-0.30, 0.30), (0.07, 0.35))
GOAL_TOLERANCE_M = 0.015
MAX_TURNS = 8
AXES = ("x_m", "y_m", "z_m")
PROMPT = (
    "You are the pick-and-place policy for a mock tabletop. Move the foam block "
    "onto X. Each turn you get the current scene and may choose one action. "
    "For travel, give the executor the final Cartesian destination in one "
    "move_ee call when possible. Trajectory generation is the executor's job. "
    "Use the scene's hover goals to align for grasping and placing."
)


def function_tool(name, description, properties):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS = [
    function_tool("move_ee", "Request a destination in mock tabletop XYZ metres.",
                  {axis: {"type": "number"} for axis in AXES}),
    function_tool("grasp", "Pick up the foam block from its hover pose.", {}),
    function_tool("place", "Set down the held foam block from the X hover pose.", {}),
]


class MockTabletop:
    def __init__(self):
        self.position = START
        self.block_state = "on_table"

    def observe(self, last_result):
        return {
            "gripper_xyz_m": self.position,
            "above_block_xyz_m": ABOVE_BLOCK,
            "above_x_xyz_m": ABOVE_X,
            "workspace_m": dict(zip(AXES, WORKSPACE)),
            "hover_tolerance_m": GOAL_TOLERANCE_M,
            "block_state": self.block_state,
            "last_result": last_result,
        }

    def apply(self, name, args):
        if not isinstance(args, dict):
            return "Expected an argument object."
        if name == "move_ee" and args.keys() == set(AXES):
            point = tuple(args[axis] for axis in AXES)
            if all(type(value) in (int, float) and low <= value <= high
                   for value, (low, high) in zip(point, WORKSPACE)):
                self.position = point
                return "Arrived at the requested mock pose."
            return "Destination is outside the mock XYZ workspace."
        if name == "grasp" and not args:
            if self.block_state == "on_table" and self.near(ABOVE_BLOCK):
                self.block_state = "held"
                return "Foam block picked up."
            return "Grasp requires an empty gripper at the block hover pose."
        if name == "place" and not args:
            if self.block_state == "held" and self.near(ABOVE_X):
                self.block_state = "on_x"
                return "Foam block placed on X."
            return "Place requires a held block at the X hover pose."
        return "Unknown action or invalid arguments."

    def near(self, goal):
        return math.dist(self.position, goal) <= GOAL_TOLERANCE_M


def request_astra(api_key, observation):
    body = {
        "model": MODEL,
        "instructions": PROMPT,
        "input": json.dumps(observation),
        "tools": TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "reasoning": {"effort": "medium"},
        "store": False,
    }
    request = Request(URL, method="POST", data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-openai-actor-authorization": "local-image-extension",
    })
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def run(api_key=None):
    scene = MockTabletop()
    report = {"source": MODEL if api_key else "scripted_smoke_test", "turns": []}
    scripted = iter([
        ("move_ee", dict(zip(AXES, ABOVE_BLOCK))), ("grasp", {}),
        ("move_ee", dict(zip(AXES, ABOVE_X))), ("place", {}),
    ])
    last_result = None
    for turn in range(1, MAX_TURNS + 1):
        observation = scene.observe(last_result)
        row = {"turn": turn, "observation": observation}
        report["turns"].append(row)
        if api_key:
            started = time.monotonic()
            try:
                response = request_astra(api_key, observation)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                row["error"] = str(exc)
                if isinstance(exc, HTTPError):
                    with exc:
                        body = exc.read(4096).decode(errors="replace")
                    row["gateway_error"] = body.replace(api_key, "[redacted]")
                break
            finally:
                row["model_seconds"] = round(time.monotonic() - started, 3)
            row["response_id"] = response.get("id")
            row["response_status"] = response.get("status")
            output = response.get("output") or []
            calls = [item for item in output if item.get("type") == "function_call"]
            if response.get("status") != "completed" or len(calls) != 1:
                row["error"] = f"Expected one tool call in a completed response, got {len(calls)}."
                row["messages"] = [item.get("content") for item in output
                                   if item.get("type") == "message"]
                break
            name = calls[0]["name"]
            try:
                args = json.loads(calls[0]["arguments"])
            except (TypeError, ValueError):
                row["error"] = "Tool arguments were not valid JSON."
                break
        else:
            name, args = next(scripted)
        last_result = scene.apply(name, args)
        row.update(action=name, arguments=args, result=last_result)
        if scene.block_state == "on_x":
            break
    report["model_calls"] = len(report["turns"]) if api_key else 0
    report["placed_on_x"] = scene.block_state == "on_x"
    report["four_action_success"] = report["placed_on_x"] and len(report["turns"]) == 4
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astra", action="store_true",
                        help="Query Astra using RHOS_API_KEY instead of scripted actions")
    args = parser.parse_args()
    api_key = os.environ.get("RHOS_API_KEY") if args.astra else None
    if args.astra and not api_key:
        parser.error("--astra requires RHOS_API_KEY in the environment")
    report = run(api_key)
    print(json.dumps(report, indent=2))
    return 0 if report["placed_on_x"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
