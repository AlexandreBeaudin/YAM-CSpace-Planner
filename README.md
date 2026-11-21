# YAM C-space planner

The original project and its research materials are in [yam-cspace-planner](yam-cspace-planner/)
and [research](research/). The [Astra harness](yam-astra-experiments/integrations/inspect_robots/)
includes the current IK planner, hover and pick-and-place prompts, replay tools, tests,
and the lab's YAM source dependencies. The harness sources come from the working tree
based on commit `1b7160a920a734f7e3fcf16c1a250dd3aa95d55a`.

## Work from a laptop

Run the mock policy from the repository root with Python 3.

```bash
python3 yam-cspace-planner/policy_probe.py
```

Open either trial's `replay.html` in a browser after cloning. Keep the replay
frames and assets next to it.

- [Trial 1](yam-cspace-planner/runs/relative-001/replay.html)
- [Trial 2](yam-cspace-planner/runs/relative-002/replay.html)

For offline IK and policy tests on macOS or Linux, use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install Pillow './yam-astra-experiments/YAM/i2rt[yam]'
cd yam-astra-experiments/integrations/inspect_robots
python -m unittest -q test_tabletop_policy test_planned_motion
```

The [station instructions](yam-astra-experiments/integrations/inspect_robots/TABLETOP_IK.md)
and [reference profile](research/station-profile.yaml) describe the lab setup.
The camera-to-arm mapping is still a calibration candidate. Both browser
replays, their image assets, and the station event logs are included. Full
raw rollout recordings and model wire captures are on the lab machine.
