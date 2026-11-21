Your existing YAM3 research stack already provides the low-level CAN bridge, safety guards, MuJoCo/Mink kinematics, and 4-lane `.mp4` replay exporter. To turn this into your **C-Space Acceleration Engine and Trajectory Smoothing** course project, you only need to build the mid-level planning/smoothing layer and wire it into `astra_run.py`.

---

## 1. Leveraging What You Already Built vs. What to Build Next

| Layer | Already Done in Your Lab Repo | New Work for Course Project |
| --- | --- | --- |
| **Low-Level Control & Safety** | `can_yam_a` follower daemon, 30 Hz interpolation, joint bounds, `0.08 rad` tracking lag guard

 | Keep completely untouched as your hard safety floor

 |
| **Kinematics & Sim** | MuJoCo IK in `integrations/inspect_robots/planned_motion.py`; Mink/Pinocchio & URDF/XML in `YAM/`

 | Ground `right_base` extrinsics to the table plane and add whole-arm link collision geometry

 |
| **Mid-Level Motion Planning** | Disabled (`astra_run.py` forces `move_joints` due to missing swept-path checker)

 | Implement 6-DOF C-Space planner (RRT-Connect / A* + Potential Field) + minimum-jerk/cubic-spline smoother |
| **Policy & Prompting** | Closed-loop `move_joints` prompts (Block V4 & USB V2)

 | Expose `move_ee(x, y, z, yaw)` tool and prompt Astra for single-leap Cartesian goals |
| **Visualization & Demo** | `replay.html` & `export_replay_mp4.py` with `Model / Tools / Motion / Images` timeline

 | Overlay planned C-Space joint curves & 3D Cartesian arc onto the exported `.mp4` |

---

## 2. Five-Milestone Project Roadmap

### Milestone 1: Ground `planned_motion.py` IK & `right_base` Frame

* **Goal:** Make forward and inverse kinematics match your real YAM3 hardware in `right_base` coordinates.


* **Deliverables:**
* Audit `integrations/inspect_robots/planned_motion.py` and verify that Forward Kinematics (FK) on your recorded Trial 13/14 joint states matches the physical gripper position over the table.


* Measure and hard-register the physical table height ($z_{\text{table}}$) and workspace bounding box in `right_base`.


* Add a posture bias in the IK solver favoring **high-elbow configurations (`j1 > j2`)** and enforcing `j3 >= -1.65 rad` so the solver never commands unreachable wrist angles like the `-1.83 rad` stall seen in USB Trial 1.





### Milestone 2: Whole-Arm C-Space Collision Checker

* **Goal:** Replace the missing `"trusted swept-path checker"` in `astra_run.py` with a fast self-collision + tabletop-obstacle collision checker.


* **Deliverables:**
* Model the table surface, left parked arm (`can_yam_b`), and tabletop obstacles as 3D primitive collision boxes/cylinders inside the MuJoCo/Pinocchio scene.


* Implement `is_config_valid(q)` and `is_path_valid(q_start, q_end, step=0.02)` checking **all arm links and gripper pads** (not just the TCP) against obstacles and table bounds.



### Milestone 3: C-Space Path Planner & Trajectory Smoother

* **Goal:** Generate collision-free, low-jerk 30 Hz joint trajectories from a single distant Cartesian target.


* **Deliverables:**
* **C-Space Planning:** Implement **RRT-Connect** (or voxelized **A*** with a repulsive obstacle potential field) in 6-DOF joint space to connect $q_{\text{current}}$ to $q_{\text{goal}} = \text{IK}(x, y, z, \text{rpy})$ around obstacles.
* **Path Shortcutting & Smoothing:** Prune redundant RRT waypoints via greedy line-of-sight shortcutting, then fit a **time-parameterized cubic/quintic spline** (or trapezoidal velocity profile) sampled at **30 Hz**. Enforce maximum joint velocity $\dot{q}_{\max}$ and acceleration $\ddot{q}_{\max}$ so the controller never triggers the `0.08 rad` tracking-lag guard.





### Milestone 4: Expose `move_ee` Tool & Aggressive Astra Prompt

* **Goal:** Let GPT-6 Astra issue single confident Cartesian leaps instead of incremental joint steps.
* **Deliverables:**
* Wire a `move_ee(x, y, z, yaw, note)` tool into `astra_run.py` that runs `IK -> C-Space Planner -> Smoother -> Swept-Path Verification` and dispatches the resulting trajectory to `can_yam_a`.


* Update the prompt to give Astra the table bounds in meters, remove step-by-step transit instructions, and instruct it to command direct pre-grasp, grasp, transfer, and release coordinates.



### Milestone 5: Benchmark Suite & Course Demo Video

* **Goal:** Collect quantitative comparisons and generate the presentation video.
* **Deliverables:**
* Run controlled trials comparing baseline `move_joints`, unsmoothed IK, and your C-Space smoothed planner.
* Extend `export_replay_mp4.py` to render a live 6-DOF joint trajectory plot ($q_0(t)\dots q_5(t)$) and 3D planned path overlay synchronized with your existing 4-lane execution timeline (`Model`, `Tools`, `Motion`, `Images`).





---

## 3. Recommended Task & Physical Environment

Do **not** use the USB-A insertion task for this course project—USB insertion is bottlenecked by sub-millimeter keyed tolerances and wrist-camera self-occlusion, which obscures the gains of your C-Space motion planner. Instead, build directly on your working pink foam block setup:

* **Primary Benchmark Task — "Obstacle-Barrier Pick-and-Place":**
* **Environment:** Keep the checkered tablecloth, the pink foam block on the right side, and the white paper marked `X` on the left side. Place a **20–25 cm tall rigid obstacle** (e.g., a taped cardboard box or foam block tower—safe if brushed during tuning) directly on the centerline between the block and the `X`.


* **Why it works:** A direct straight-line joint or Cartesian move from the block to the `X` collides head-on with the central barrier. Your C-Space planner must compute a smooth curved arc up and over (or around) the barrier from a **single** `move_ee` call above the `X`.


* **Secondary Stretch Scenario — "Shelf / Box Reach-In" (Matches Proposal Q7):**
* Place the white `X` paper inside a wide, open-front cardboard box ("drawer/cubby") so the top is covered.
* When Astra commands a placement pose inside the cubby, your planner must route the arm horizontally through the open front face without colliding the forearm (`j2`–`j3` link) against the top lip of the box—highlighting full-body C-Space collision checking rather than simple TCP clearance.



---

## 4. Experiments & Metrics to Include in Your Final Report

Run **5 trials per condition** on the Obstacle-Barrier Pick-and-Place task:

| Condition | Control Interface | Planner / Smoother | Purpose |
| --- | --- | --- | --- |
| **1. Joint-Space Baseline** | `move_joints` (`0.80 rad` cap)

 | None (Astra manually steps around obstacle) | Measures baseline LLM call count (~15–25 calls) and ~10–15 min runtime

 |
| **2. Naive Cartesian IK** | `move_ee` (direct linear interpolation) | IK only, no obstacle avoidance | Demonstrates failure mode (collides with barrier or triggers safety stop) |
| **3. Unsmoothed C-Space RRT** | `move_ee` | RRT-Connect without spline smoothing | Ablation isolating how jerky piecewise-linear paths cause tracking lag (`> 0.08 rad`)

 |
| **4. Full C-Space Engine (Ours)** | `move_ee` + Aggressive Prompt | IK + RRT-Connect + Shortcutting + Quintic Spline | Target system: 4–6 total LLM calls, zero collisions, smooth velocity profiles |

**Key Quantitative Plots & Analyses for Your Report:**

1. **Inference & Wall-Clock Speedup:** Bar chart comparing total LLM calls and total seconds (split into **Model Inference Time** vs. **Physical Motion Time** from your replay JSONL) across Conditions 1 and 4.
2. **Trajectory Smoothness & Tracking Fidelity:** Plot joint positions $q_i(t)$, velocities $\dot{q}_i(t)$, and **mean squared jerk** $\frac{1}{T}\int_0^T \Vert{}\dddot{q}(t)\Vert{}^2 dt$, alongside peak encoder tracking residual (`|target - measured|`) for Condition 3 vs. Condition 4.


3. **Trajectory Diversity Analysis (Proposal Q6):** Overlay 5–10 RRT-Connect C-Space solutions for the same start/goal pair in 3D workspace to illustrate sampling-based path diversity and how shortcutting converges them toward near-optimal arcs.
4. **Prompt Aggressiveness Ablation:** Compare conservative vs. aggressive system prompts on `move_ee` to show how many intermediate "safety waypoints" GPT-6 Astra inserts on its own before trusting the mid-level planner to handle obstacle avoidance in one leap.

---

### Immediate Concrete Next Step

Run a read-only inspection of `integrations/inspect_robots/planned_motion.py` and your Trial 13/14 joint logs in your active Codex CLI session to verify the exact FK transform from `right_base` to the gripper TCP:

```bash
YAM/gello_software/.venv/bin/python -c "import inspect, integrations.inspect_robots.planned_motion as pm; print(inspect.getsource(pm))"

```