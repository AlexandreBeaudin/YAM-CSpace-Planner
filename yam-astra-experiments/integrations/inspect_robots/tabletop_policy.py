"""Cartesian pick-and-place tools for the station policy."""

import numpy as np

from inspect_robots import Action, ActionChunk
from inspect_robots.tools import Tool, ToolLibrary, ToolResult
from planned_motion import PosePlanner
from tabletop_targets import (
    JAW_LINE_SCHEMA, action_segments, grasp_settings, plan_action, project_targets,
    top_down_grasp_rotation,
)


DESCRIPTIONS = {
    'move_above_block': 'Travel directly to a hover above the foam block with the selected jaw angle, or adjust the hover.',
    'grasp': 'From the block hover, lower, close and lift at the selected jaw angle. Use the latest view to confirm alignment and that the block is on the table.',
    'move_above_x': 'With the block visibly held, travel directly to a hover above X while preserving the grasp angle.',
    'place': 'From the X hover with the block visibly held, lower, release and retract.',
}


class TabletopPolicyTools:
    def __init__(self, calibration, setup, clearance_check=None):
        if calibration.get('main_preview_matches_calibration') is not True:
            raise ValueError('validate the calibration against the exact live main preview image geometry')
        self.calibration = calibration
        self.arm_side = calibration.get('arm_side', 'right')
        self.arm_type = setup['arm_type']
        self.settings = grasp_settings(calibration, setup)
        self.planner = PosePlanner(arm_type=self.arm_type)
        self.clearance_check = clearance_check
        project_targets(calibration, {'x_center': {'u': .5, 'v': .5}}, (640, 480))

    def tools(self, arm, max_steps):
        if not callable(self.clearance_check):
            return ToolLibrary([])
        tools = []
        for name, description in DESCRIPTIONS.items():
            properties = {
                'u': {'type': 'number'}, 'v': {'type': 'number'},
                'note': {'type': 'string'},
            }
            if name == 'move_above_block':
                properties['jaw_line'] = JAW_LINE_SCHEMA
            if name == 'grasp':
                properties['block_on_table'] = {'type': 'boolean'}
            if name in {'move_above_x', 'place'}:
                properties['block_visibly_held'] = {'type': 'boolean'}
            tools.append(Tool(name, {'type': 'function', 'function': {
                'name': name, 'description': description + ' Supply u and v for the target center in the current main image.',
                'parameters': {'type': 'object', 'properties': properties,
                               'required': list(properties), 'additionalProperties': False},
            }}, lambda args, obs, name=name: self.execute(arm, max_steps, name, args, obs), 'control'))
        return ToolLibrary(tools)

    def execute(self, arm, max_steps, name, args, observation):
        arm.check()
        previous = arm.tabletop_last_action
        start = arm.last.copy()
        measured = np.asarray(observation.state['joint_pos'])
        at_block = name in {'move_above_block', 'grasp'}
        target_name = 'block_top' if at_block else 'x_center'
        hover_name = 'above_block_m' if at_block else 'above_x_m'

        try:
            if not isinstance(args.get('note'), str) or not args['note'].strip():
                raise ValueError('describe the current visual evidence in note')
            if previous == 'place':
                raise ValueError('inspect the placement and finish the trial')
            if name == 'move_above_block' and (previous == 'move_above_x' or measured[6] < .95):
                raise ValueError('approaching the block requires an open gripper and no transfer in progress')
            if name == 'grasp' and (previous not in {'move_above_block', 'grasp'}
                                    or args.get('block_on_table') is not True):
                raise ValueError('grasp requires confirmed block-on-table evidence from the block hover')
            if not at_block and (previous not in {'grasp', 'move_above_x'}
                                 or args.get('block_visibly_held') is not True):
                raise ValueError('transfer and place require visual confirmation that the block is held')
            if name == 'place' and previous != 'move_above_x':
                raise ValueError('move to the hover above X before placing')

            image = observation.images['main']
            pixels = {target_name: {'u': args['u'], 'v': args['v']}}
            jaw_line = args['jaw_line'] if name == 'move_above_block' else None
            if name == 'move_above_block' and jaw_line is None:
                raise ValueError('choose the jaw line from the current main image')
            image_size = (image.shape[1], image.shape[0])
            targets = project_targets(self.calibration, pixels, image_size, jaw_line=jaw_line)
            base = targets['frame']
            rotation = arm.tabletop_grasp_rotation
            if name == 'move_above_block':
                rotation = top_down_grasp_rotation(targets[f'jaw_direction_{base}'],
                                                   self.calibration[f'table_normal_{base}'],
                                                   self.planner.fk(measured)[1])
            if rotation is None:
                raise ValueError('select a grasp angle at the block hover first')
            if name in {'grasp', 'place'}:
                position, actual_rotation = self.planner.fk(measured)
                if (np.linalg.norm(position - targets[hover_name]) > .015
                        or np.linalg.norm(actual_rotation - rotation) > .08):
                    raise ValueError('measured gripper is off the current hover pose; adjust the hover first')
            segments = action_segments(name, targets, self.settings,
                                       open_before_grasp=(name == 'grasp' and start[6] < 1.))
            rows = plan_action(self.planner, name, segments, rotation, start, arm.space.low, arm.space.high)
            points = np.concatenate([row for _, row in rows])
            if arm.t + len(points) > max_steps:
                raise ValueError('the complete action would exceed the remaining control-step budget')
            context = {
                'action': name, 'targets': targets, 'pixels': pixels, 'note': args['note'],
                'jaw_line': jaw_line, f'target_rotation_{base}': rotation.tolist(),
                'active_arm': arm.active_side, 'previous_action': previous,
                f'initial_{arm.parked_side}_joints': arm.initial[arm.parked_slice].tolist(),
                f'measured_{arm.active_side}_joints': measured.tolist(),
                'segments': [{'name': segment, 'waypoints': len(row)} for segment, row in rows],
                'camera_metadata': observation.extra.get('cameras', {}),
            }
            arm.record(dict(kind='tabletop_plan', step=arm.t, **context, waypoints=points.tolist()))
            def check(path):
                return self.clearance_check(path, context, observation)
            if rejection := arm.pre_check(points, planned=True, planned_checker=check):
                raise ValueError(rejection)
        except (KeyError, TypeError, ValueError) as exc:
            arm.record(dict(kind='tabletop_rejected', step=arm.t, action=name, reason=str(exc)))
            return ToolResult(error=str(exc))
        actions = [Action(data=point.copy()) for point in points]
        meta = {'chunk_final': True, 'tabletop_action': name}
        if name == 'move_above_block':
            meta['tabletop_grasp_rotation'] = rotation.tolist()
        actions[-1] = Action(data=points[-1].copy(), meta=meta)
        arm.record(dict(kind='tabletop_approved', step=arm.t, action=name, waypoint_count=len(points)))
        return ToolResult(chunk=ActionChunk(actions=actions, control_hz=30.), target=points[-1].copy(),
                          note=f'{name} approved to execute over {len(points)} steps. Inspect the next images and readback.')
