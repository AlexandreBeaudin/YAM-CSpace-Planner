"""Bounded relative end-effector tools for a visually guided tabletop trial."""

import numpy as np

from inspect_robots import Action, ActionChunk
from inspect_robots.tools import Tool, ToolLibrary, ToolResult
from planned_motion import checked_vector, trajectory


MOVE_LIMITS = {
    'near_object': (.05, .25),
    'clear_transit': (.20, 1.0),
}


def relative_tools(arm, max_steps):
    def execute(args, observation, *, gripper=False):
        arm.check()
        try:
            if not isinstance(args.get('note'), str) or not args['note'].strip():
                raise ValueError('describe the visual evidence for this move')
            start = arm.last.copy()
            measured = checked_vector(arm.measured_feedback()['position'], (7,))
            target = start.copy()
            if gripper:
                opening = args.get('opening')
                if type(opening) not in (int, float) or not 0 <= opening <= 1:
                    raise ValueError('opening must be between 0 and 1')
                target[6] = opening
            else:
                phase = args.get('phase', 'near_object')
                if not isinstance(phase, str) or phase not in MOVE_LIMITS:
                    raise ValueError('phase must be near_object or clear_transit')
                max_translation, max_rotation = MOVE_LIMITS[phase]
                delta = checked_vector(args['delta_m'], (3,))
                rotation_delta = checked_vector(args['rotation_vector_rad'], (3,))
                angle = np.linalg.norm(rotation_delta)
                if np.linalg.norm(delta) > max_translation + 1e-9 or angle > max_rotation + 1e-9:
                    raise ValueError(f'{phase} allows up to {max_translation:g} m translation and '
                                     f'{max_rotation:g} rad rotation per call')
                position, rotation = arm.pose_planner.fk(measured)
                if angle:
                    x, y, z = rotation_delta / angle
                    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    rotation = (np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)) @ rotation
                target = arm.pose_planner.solve(start, position + delta, rotation, arm.space.low, arm.space.high)
            points = trajectory(start, target)
            if arm.t + len(points) > max_steps:
                raise ValueError('move exceeds the remaining step budget')
            if rejection := arm.pre_check(points):
                raise ValueError(rejection)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return ToolResult(error=str(exc))
        arm.record(dict(kind='relative_eef_plan', step=arm.t, action='set_gripper' if gripper else 'move_eef_by',
                        arguments=args, measured_start=measured.tolist(), waypoints=points.tolist()))
        actions = [Action(data=point.copy()) for point in points]
        actions[-1] = Action(data=points[-1].copy(), meta={'chunk_final': True, 'tabletop_action':
                                                       'set_gripper' if gripper else 'move_eef_by'})
        return ToolResult(chunk=ActionChunk(actions=actions, control_hz=30.), target=target.copy(),
                          note=f'Executing over {len(points)} steps. Check the new view and readback.')

    vector = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}

    def tool(name, description, properties, *, gripper=False, optional=()):
        properties = {**properties, 'note': {'type': 'string'}}
        return Tool(name, {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': [key for key in properties if key not in optional],
                           'additionalProperties': False}}},
            lambda args, obs: execute(args, obs, gripper=gripper), 'control')

    return ToolLibrary([
        tool('move_eef_by', 'Translate and rotate the grasp point relative to its measured pose. '
             'Both vectors use the left arm base axes in XYZ order. Positive Z raises the tool. '
             'Use rotation_vector_rad = [0,0,0] to keep orientation. The vector direction is the '
             'right-hand rotation axis and its length is the angle. '
             'Use clear_transit only when the views show free space along the move. '
             'Use near_object for descent, final alignment, or work close to the table or block.',
             {'delta_m': vector, 'rotation_vector_rad': vector,
              'phase': {'type': 'string', 'enum': list(MOVE_LIMITS),
                        'description': 'clear_transit allows 0.20 m and 1.0 rad. near_object is the default '
                                       'and allows 0.05 m and 0.25 rad.'}}, optional=('phase',)),
        tool('set_gripper', 'Set gripper opening independently. 1 is open and 0 is closed.',
             {'opening': {'type': 'number', 'minimum': 0, 'maximum': 1}}, gripper=True),
    ])
