# YAM C-space planner

## Astra pick-and-place policy test

The first experiment measures whether Astra chooses four high-level actions
for a foam-block transfer. The expected calls are `move_ee` above the foam
block, `grasp`, `move_ee` above X, and `place`. Astra receives the same
three tools and the current scene on each turn. It can choose intermediate
destinations anywhere in the mock XYZ workspace.

The tabletop coordinates are synthetic. This test uses a position and block
state model with ideal arrival, grasp, and release. It measures policy decisions
with known target positions. A physical trial requires calibrated targets and
collision-checked arm trajectories, including the grasp and place motions.

From this folder, run the scripted smoke test with Python 3.8 or later.

```sh
python3 policy_probe.py
```

To query the lab's RHOS gateway with `RHOS_API_KEY` set in the environment, run:

```sh
python3 policy_probe.py --astra
```

The JSON report includes the action arguments, mock outcome, request timings,
and whether the policy placed the block in exactly four actions. The scripted
smoke test reports zero model calls.
