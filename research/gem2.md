### 1. Extracting 3D Bounding Boxes: VGGT vs. SpatialHarness

For a semester robotics project, **VGGT (Visual Geometry Grounded Transformer)** is a much better fit than reproducing full SpatialHarness:

* **Why skip full SpatialHarness:** Reproducing an end-to-end VLM-driven Real2Sim scene generator introduces heavy failure modes (mesh hallucinations, scale drift, long generation times) that can stall your project before you even test your C-Space planner.
* **How to use VGGT cleanly (Step-0 Static Snapshot):**
1. At `step 0` (before the right arm moves), pass the synchronized RGB frames from `main`, `left`, and `right` into VGGT once to generate a dense 3D point cloud.
2. **Metric Alignment to `right_base`:** Because VGGT outputs up-to-scale camera-frame geometry, register the point cloud into `right_base` using known rigid landmarks visible in `main` (specifically the aluminum extrusion rail/pedestal and the parked left arm's known FK joint pose).


3. **Voxel / AABB Extraction:** Crop out the table plane ($z > z_{\text{table}} + 1\text{ cm}$), cluster the remaining obstacle points (e.g., DBSCAN or connected-component voxels), and convert each cluster into an Axis-Aligned Bounding Box (AABB) inflated by a `2–3 cm` safety margin. Feed those AABBs directly into your MuJoCo/Pinocchio collision checker.




* **Experimental Best Practice:** Support **two obstacle input modes** in your planner—(a) **known primitive boxes** (for repeatable, deterministic C-Space smoothness/jerk benchmarks) and (b) **VGGT-estimated AABBs** (to demonstrate zero-calibration multi-view perception-to-planning).

---

### 2. Harness Conservatism vs. GPT-6 Alignment (And the Code-as-Policy Pivot)

Jiaxin’s suspicion about the existing harness injecting safety conservatism is **100% confirmed by your telemetry logs**:

* **What your harness is actually doing right now:** Even after we rewrote your user task prompt, look at `shared_decision_report_lines` and `decision_report_lines` inside `USB-A 82-min joint telemetry.jsonl`. On **every single turn**, the runner injects lines like:


* *"Large coupled moves are for clear free-space transit only. Near the hub, align keyed plug orientation first and use small observed moves..."*


* *"Tolerance is not proof of clearance or contact; never chase a residual."*


* *"Content change alone is not exposure freshness or useful motion."*




* **Why stripping the harness is a major finding:** In a minimal harness (keeping only low-level CAN joint limits, velocity limits, and tracking-lag guards), you can run a clean ablation comparing **Stock Harness** vs. **Minimal Harness** to measure how much of GPT-6 Astra's hesitation comes from harness prompt injection versus the model's own post-training alignment.


* **Why Code-as-Policy is a win-win fallback (not a failure):**
* If Astra **does** output single distant Cartesian targets (`move_ee(x, y, z, yaw)`) in the minimal harness, your C-Space smoother executes the curved obstacle-avoiding swoop in 1 call.
* If Astra **still** hedges and breaks distant moves into multiple intermediate `move_ee` waypoints due to post-training preferences, letting Astra output a short **Code-as-Policy Python snippet** (e.g., `plan_and_move(above_block); grasp(); plan_and_move(above_x); release()`)—which your C-Space engine validates and executes as a continuous multi-segment trajectory—cuts an entire pick-and-place episode down to **1–2 API calls** while keeping your C-Space planner at the center of the architecture.



---

### Immediate Execution Checklist

1. **Inspect `planned_motion.py` & `astra_run.py` Harness Injections:**
In `/home/rhos/workspace/jiaxin/shurui`, inspect `integrations/inspect_robots/planned_motion.py` and search `astra_run.py` for `shared_decision_report_lines` / `decision_report_lines` so you know exactly how the existing MuJoCo IK solver works and where the per-turn warning lines are injected.


2. **Build the Standalone C-Space Planner & Smoother Module (`cspace_planner.py`):**
Implement whole-arm link collision checking against table + obstacle AABBs, RRT-Connect path planning in 6-DOF joint space, greedy path shortcutting, and 30 Hz quintic-spline trajectory smoothing.


3. **Wire `move_ee` into a Minimal-Harness Mode (`--minimal-harness`):**
Add a flag in `astra_run.py` that disables the per-turn residual/caution boilerplate and exposes `move_ee(x, y, z, yaw)` backed by your C-Space planner, while preserving all low-level `can_yam_a` safety guards.


4. **Run the 3-Condition Benchmark on the Obstacle-Barrier Task:**
Compare (1) Stock `move_joints` baseline, (2) Minimal-Harness `move_ee` with C-Space smoothing, and (3) VGGT-reconstructed obstacle avoidance (plus Code-as-Policy multi-step execution if Astra still subdivides Cartesian leaps), logging LLM calls, wall-clock latency, peak tracking residual, and trajectory jerk.