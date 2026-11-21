"""Closed-loop Cartesian policy through the real harness with synthetic I/O."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import astra_run as run
import httpx
from demo import offline_only
from inspect_robots import Scene
from planned_motion import PosePlanner
from tabletop_targets import project_targets, top_down_grasp_rotation
from test_astra_run import FakePreview, FakeRobot


def synthetic_station(checker, side='right'):
    class BimanualRobot(FakeRobot):
        def get_diagnostics(self):
            record = super().get_diagnostics()
            record['followers']['left'] = {
                **record['followers']['right'], 'position': self.joints[:7].copy()}
            return record

    robot = BimanualRobot()
    nominal = np.array([.32224, 1.16006, 1.06006, -1.47047, -.00109, -.36914, 1.])
    start = 7 if side == 'right' else 0
    robot.joints[start:start+7] = nominal
    robot.joints[start:start+2] -= [.02, .01]
    position, rotation = PosePlanner(arm_type='yam').fk(nominal)
    x, y, z = position
    calibration = {
        'arm_side': side,
        'main_preview_matches_calibration': True,
        'main_size_px': [640, 480], 'main_intrinsics_px': [500, 500, 319.5, 239.5],
        f'{side}_base_T_main_optical': [[1, 0, 0, x], [0, -1, 0, y],
                                      [0, 0, -1, z + .50], [0, 0, 0, 1]],
        f'table_point_{side}_base_m': [0, 0, z - .06], f'table_normal_{side}_base': [0, 0, 1],
        'block_height_m': .03, 'hover_clearance_m': .03,
    }
    setup = {'arm_type': 'yam',
             'grasp_depth_below_top_m': .015, 'release_clearance_m': .002}
    tabletop = run.TabletopPolicyTools(calibration, setup, checker)
    arm = run.StationArm(robot, FakePreview(robot), {'joint_limits': [[-3., 3.]] * 14},
                         simulated=True, task_id='tabletop-ik', tabletop=tabletop)
    delta_u = .026 * float(rotation[0, 1])
    delta_v = -.035 * float(rotation[1, 1])
    jaw_line = {'start': {'u': .5 - delta_u, 'v': .5 - delta_v},
                'end': {'u': .5 + delta_u, 'v': .5 + delta_v}}
    return arm, robot, jaw_line


class TabletopPolicyTests(unittest.TestCase):
    def test_relative_trial_exposes_ik_and_gripper_tools_without_calibration(self):
        fixture, robot, _ = synthetic_station(None, side='left')
        arm = run.StationArm(robot, fixture.preview, {'joint_limits': [[-3., 3.]] * 14},
                             simulated=True, task_id='tabletop-relative',
                             pose_planner=fixture.tabletop.planner)
        spec = run.task_configuration('tabletop-relative')
        transit = dict(delta_m=[.12, 0., 0.], rotation_vector_rad=[0., 0., .40])
        observation = arm.reset(Scene(id='relative-limits', instruction='Use left arm'))
        result = run.relative_tools(arm, run.MAX_STEPS).execute(
            'move_eef_by', json.dumps(dict(transit, note='Synthetic near-object request')), observation)
        self.assertIn('near_object allows up to', result.error)
        self.assertFalse(robot.commands)
        calls = [
            ('move_eef_by', dict(transit, phase='clear_transit')),
            ('set_gripper', dict(opening=0.95)),
            ('set_gripper', dict(opening=1.0)),
            ('done', dict(summary='Synthetic relative trial complete')),
        ]
        initial_right = robot.joints[7:].copy()
        position, rotation = arm.pose_planner.fk(robot.joints[:7])
        readback = robot.get_diagnostics
        turn = 0

        def measured_with_elbow_offset():
            diagnostics = readback()
            diagnostics['followers']['left']['velocity'] = [0.] * 7
            diagnostics['followers']['left']['feedback_effort'] = [0.] * 7
            if robot.commands:
                diagnostics['followers']['left']['position'][2] -= .0431
            return diagnostics

        def respond(request):
            nonlocal turn
            body = json.loads(request.content)
            names = {tool['name'] for tool in body['tools']}
            self.assertTrue({'move_eef_by', 'set_gripper'} <= names)
            self.assertNotIn('move_joints', names)
            self.assertIn('Measured grasp point from FK in left-base axes', json.dumps(body))
            if turn == 1:
                moved, rotated = arm.pose_planner.fk(robot.joints[:7])
                np.testing.assert_allclose(moved - position, transit['delta_m'], atol=.0005)
                expected_turn = 2 * np.sqrt(2) * np.sin(.40 / 2)
                self.assertAlmostEqual(np.linalg.norm(rotated - rotation), expected_turn, delta=.002)
                self.assertIn('Measured grasp-point displacement from FK', json.dumps(body))
                self.assertIn('arm residual 0.04310 rad', json.dumps(body))
            if turn == 2:
                self.assertAlmostEqual(robot.joints[6], .95)
            name, args = calls[turn]
            turn += 1
            return httpx.Response(200, json={'status': 'completed', 'output': [{
                'type': 'function_call', 'call_id': str(turn), 'name': name,
                'arguments': json.dumps({'note': 'Synthetic evidence', **args})}]})

        transport = run.GatewayTransport(httpx.MockTransport(respond))
        policy = run.make_policy(transport, arm.pre_check, station=arm,
                                 env={'RHOS_API_KEY': 'offline-placeholder'}, prompt_path=spec['prompt'])
        with (tempfile.TemporaryDirectory() as output, offline_only(), patch.object(run.time, 'sleep'),
              patch.object(robot, 'get_diagnostics', measured_with_elbow_offset)):
            log = run.execute(policy, arm, Path(output), task_spec=spec)
            self.assertEqual(log.status, 'success', log.to_dict()['samples'][0].get('error'))
            self.assertEqual(turn, len(calls))
            self.assertAlmostEqual(robot.joints[6], 1.)
            self.assertTrue(all(np.array_equal(q[7:], initial_right) for q in robot.commands))
            recorded = next(event['motor_feedback'] for event in arm.events
                            if event['kind'] == 'observation' and event['step'] > 0)
            self.assertEqual(recorded['feedback_effort'], [0.] * 7)
            self.assertIsNone(arm.pre_check([arm.last.copy()]))
            lagged = arm.last.copy()
            lagged[2] -= .081
            count = len(robot.commands)
            feedback = {'position': lagged.tolist(), 'source_time_ns': 10,
                        'velocity': [0.] * 7, 'feedback_effort': [0.] * 7}
            with patch.object(arm, 'measured_feedback', return_value=feedback):
                with self.assertRaisesRegex(run.SafetyAbort, '0.08 rad during motion'):
                    arm.step(run.Action(data=arm.last.copy()))
            self.assertEqual(len(robot.commands), count)
            abort = arm.events[-1]
            self.assertEqual(abort['joint'], 'left_j2')
            self.assertAlmostEqual(abort['joint_errors_rad'][2], .081)
            self.assertEqual(abort['motor_feedback'], feedback)

    def test_left_arm_path_uses_left_feedback_and_keeps_right_parked(self):
        def check(path, context, observation):
            self.assertEqual(context['active_arm'], 'left')
            self.assertEqual(context['targets']['frame'], 'left_base')
            self.assertIn('target_rotation_left_base', context)
            np.testing.assert_array_equal(path[0], observation.state['joint_pos'])
            return True

        arm, robot, jaw_line = synthetic_station(check, side='left')
        robot.joints[:7] = [.3, .50, .40, -.60, 0., -.36, 1.]
        observation = arm.reset(Scene(id='left', instruction='Use left arm'))
        initial_right = robot.joints[7:].copy()
        np.testing.assert_array_equal(observation.state['joint_pos'], robot.joints[:7])
        result = arm.tabletop.tools(arm, run.MAX_STEPS).execute('move_above_block',
            json.dumps({'u': .5, 'v': .5, 'jaw_line': jaw_line, 'note': 'Synthetic target'}), observation)
        self.assertIsNone(result.error)
        self.assertGreater(np.max(np.abs(result.target[:6] - observation.state['joint_pos'][:6])), .8)
        with patch.object(run.time, 'sleep'):
            for action in result.chunk.actions:
                arm.step(action)
        np.testing.assert_array_equal(robot.joints[:7], result.target)
        self.assertTrue(all(np.array_equal(q[7:], initial_right) for q in robot.commands))
        robot.joints[7] += .06
        with self.assertRaisesRegex(RuntimeError, 'right arm/gripper drifted'):
            arm.measured_feedback()

    def test_decisions_use_new_feedback_after_each_checked_action(self):
        checked = []
        rotations = []

        def check(path, context, observation):
            self.assertFalse(path.flags.writeable)
            self.assertEqual(observation.images['main'].shape, (480, 640, 3))
            checked.append((context['action'], path.copy()))
            target_rotation = np.asarray(context['target_rotation_right_base'])
            rotations.append(target_rotation)
            np.testing.assert_allclose(target_rotation[:, 2], [0, 0, -1], atol=1e-8)
            np.testing.assert_allclose(arm.tabletop.planner.fk(path[-1])[1], target_rotation, atol=.002)
            if context['action'] == 'move_above_block':
                direction = context['targets']['jaw_direction_right_base']
                self.assertAlmostEqual(abs(target_rotation[:, 1] @ direction), 1.)
                planner = arm.tabletop.planner
                slide = planner.mj.mj_name2id(planner.model, planner.mj.mjtObj.mjOBJ_JOINT, 'joint7')
                self.assertAlmostEqual(abs(planner.data.xaxis[slide] @ direction), 1., places=5)
            return True

        arm, robot, jaw_line = synthetic_station(check)
        spec = run.task_configuration('tabletop-ik')
        expected = ['move_above_block', 'grasp', 'move_above_x', 'place']
        requests = []

        def respond(request):
            body = json.loads(request.content)
            turn = len(requests)
            requests.append(body)
            names = {tool['name'] for tool in body['tools']}
            self.assertTrue(set(expected) <= names)
            self.assertNotIn('move_joints', names)
            self.assertNotIn('set_gripper', names)
            self.assertEqual(arm.tabletop_last_action, expected[turn - 1] if turn else None)
            self.assertEqual(len(checked), turn)
            self.assertGreaterEqual(json.dumps(body).count('input_image'), 3)
            if turn:
                self.assertIn('Measured change since previous decision', json.dumps(body))
                np.testing.assert_allclose(robot.joints[7:], checked[-1][1][-1])
            if turn == 4:
                replies = [('done', {'summary': 'Synthetic placement verified'})]
            else:
                replies = [(expected[turn], {'u': .5 if turn < 2 else .54, 'v': .5,
                            'note': 'Synthetic observation', **(
                                {'block_on_table': True} if turn == 1 else
                                {'block_visibly_held': True} if turn >= 2 else {'jaw_line': jaw_line})})]
                if turn == 0:
                    replies.append(('grasp', {'u': .5, 'v': .5, 'note': 'Unexpected batched call',
                                             'block_on_table': True}))
            return httpx.Response(200, json={'status': 'completed', 'output': [
                {'type': 'function_call', 'call_id': f'{turn}-{index}', 'name': name,
                 'arguments': json.dumps(args)} for index, (name, args) in enumerate(replies)]})

        transport = run.GatewayTransport(httpx.MockTransport(respond), timing_sink=arm.record)
        policy = run.make_policy(transport, arm.pre_check, station=arm,
                                 env={'RHOS_API_KEY': 'offline-placeholder'}, prompt_path=spec['prompt'])
        evidence = os.environ.get('YAM_SYNTHETIC_RUN_DIR')
        with tempfile.TemporaryDirectory() as output, offline_only(), patch.object(run.time, 'sleep'):
            output = Path(evidence or output)
            if evidence:
                output.mkdir(parents=True, exist_ok=False)
            arm.journal = run.EvidenceJournal(output)
            recorder = run.PreviewRecorder(arm.preview, arm.journal)
            recorder.capture()
            log = run.execute(policy, arm, output, task_spec=spec)
            recorder.capture()
            if log.status != 'success':
                sample = log.to_dict()['samples'][0]
                self.fail(str({key: sample[key] for key in sample if 'error' in key or 'terminat' in key}))
            replay = run.export_replay(output)
            self.assertTrue(replay.exists())
            payload = json.JSONDecoder().raw_decode(replay.read_text().split('const data=', 1)[1])[0]
            timeline = payload['presentation']
            self.assertTrue(timeline['main'] and timeline['right'] and timeline['motion'])
            self.assertEqual(len(timeline['decisions']), 5)
            accepted = [tool for tool in timeline['tools'] if tool['steps']]
            self.assertEqual([tool['name'] for tool in accepted], expected)
        self.assertEqual(log.status, 'success')
        self.assertEqual([name for name, _ in checked], expected)
        for rotation in rotations[1:]:
            np.testing.assert_array_equal(rotation, rotations[0])
        self.assertEqual(len(requests), 5)
        self.assertEqual(len(robot.commands), sum(len(path) - 1 for _, path in checked))

    def test_unavailable_or_rejected_checker_prevents_motion(self):
        arm, robot, jaw_line = synthetic_station(None)
        obs = arm.reset(Scene(id='tabletop-ik', instruction='Mock tabletop'))
        self.assertFalse(list(arm.tabletop.tools(arm, run.MAX_STEPS)))
        arm.tabletop.clearance_check = lambda *args: False
        result = arm.tabletop.tools(arm, run.MAX_STEPS).execute(
            'move_above_block', json.dumps({'u': .5, 'v': .5, 'jaw_line': jaw_line,
                                          'note': 'Synthetic observation'}), obs)
        self.assertIn('whole-path clearance', result.error)
        self.assertIsNone(arm.tabletop_grasp_rotation)
        self.assertFalse(robot.commands)

    def test_hover_only_takes_one_model_call_and_records_arrival(self):
        arm, robot, jaw_line = synthetic_station(lambda *args: True)
        arm.hover_only = True
        spec = run.task_configuration('tabletop-ik')
        spec['prompt'] = spec['prompt'].with_name('hover_prompt.txt')
        spec['instruction'] = 'Move the open right gripper above the block.'

        def respond(request):
            names = {tool['name'] for tool in json.loads(request.content)['tools']}
            self.assertIn('move_above_block', names)
            self.assertFalse({'grasp', 'move_above_x', 'place'} & names)
            return httpx.Response(200, json={'status': 'completed', 'output': [{
                'type': 'function_call', 'call_id': 'hover', 'name': 'move_above_block',
                'arguments': json.dumps({'u': .5, 'v': .5, 'jaw_line': jaw_line,
                                          'note': 'Synthetic block center and jaw line'})}]})

        transport = run.GatewayTransport(httpx.MockTransport(respond))
        policy = run.make_policy(transport, arm.pre_check, station=arm,
                                 env={'RHOS_API_KEY': 'offline-placeholder'}, prompt_path=spec['prompt'])
        with tempfile.TemporaryDirectory() as output, offline_only(), patch.object(run.time, 'sleep'):
            log = run.execute(policy, arm, Path(output), task_spec=spec)
        self.assertEqual(log.status, 'success', log.to_dict().get('error'))
        self.assertEqual(transport.calls, 1)
        self.assertEqual(arm.tabletop_last_action, 'move_above_block')
        self.assertTrue(all(target[13] == 1. for target in robot.commands))
        self.assertEqual([event['step'] for event in arm.events if event['kind'] == 'decision_observation'],
                         [0, len(robot.commands)])

    def test_projected_jaw_line_controls_rotation_and_requires_distinct_points(self):
        arm, _, jaw_line = synthetic_station(None)
        calibration = arm.tabletop.calibration
        center = {'block_top': {'u': .5, 'v': .5}}
        with self.assertRaisesRegex(ValueError, 'too close'):
            project_targets(calibration, center, (640, 480),
                            jaw_line=dict.fromkeys(('start', 'end'), jaw_line['start']))
        current = arm.tabletop.planner.fk(arm.robot.joints[7:])[1]
        direction = project_targets(calibration, center, (640, 480), jaw_line=jaw_line)['jaw_direction_right_base']
        normal = calibration['table_normal_right_base']
        rotation = top_down_grasp_rotation(direction, normal, current)
        reverse = top_down_grasp_rotation(-np.asarray(direction), normal, current)
        np.testing.assert_allclose(rotation, reverse)
        perpendicular = np.cross(normal, direction)
        alternate = top_down_grasp_rotation(perpendicular, normal, current)
        self.assertAlmostEqual(float(rotation[:, 1] @ alternate[:, 1]), 0.)
