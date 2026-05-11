# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ROS 2 Workspace Context

This is a ROS 2 (`ament_python`) package living inside a colcon workspace at `~/phd/software/ros_workspaces/test_ws`. All colcon commands must be run from the workspace root, not from this package directory.

```bash
# Build this package
cd ~/phd/software/ros_workspaces/test_ws
colcon build --packages-select ur_data_collection

# Source after building
source install/setup.bash

# Run the collector node
ros2 run ur_data_collection collector

# Run tests (flake8, pep257, copyright)
colcon test --packages-select ur_data_collection
colcon test-result --verbose
```

> **Note:** `COLCON_IGNORE` is present at the package root, which prevents colcon from discovering this package. Remove it to enable building.

## Architecture

The package implements a single ROS 2 node (`CollectorNode` in `ur_data_collection/collector.py`) that records synchronized robot manipulation data for offline learning.

**Data flow:**
- Subscribes to `/tcp_pose_broadcaster/pose` (PoseStamped), `/joint_states`, camera image topics, and an object mask topic
- All poses are transformed into `world` frame via TF2 before storage
- A timer callback at configurable Hz assembles a snapshot of all latest values and appends to `recorded_data` (a `defaultdict(list)`)
- On shutdown, the full `recorded_data` dict is pickled to `saved_data/data_<datetime>.pkl`

**Recorded fields per timestep:** `img`, `q`, `qd`, `eff`, `ee_pos`, `ee_quat_wxyz`, `ee_vel`, `gripper_pos`, per-object keypoint positions, `action`, `reward`, `termination`

**External dependencies (not in package.xml):**
- `goc_mpc` — GoC-MPC planner (`Block`, `GraphOfConstraints`, `GraphOfConstraintsMPC`, `SimpleDrakeGym`)
- `goc_demo` — task plan builders (`one_robot_move_in_circles_builder`, `pick_and_place_builder`) and `robotiq` gripper driver
- `pydrake` — math utilities (`RollPitchYaw`, `Quaternion`)

**Hardware assumptions:**
- Robotiq gripper hardcoded at IP `10.168.4.249`, port `63352` — instantiated unconditionally in `__init__`
- `saved_data/` directory must exist before running; the node does not create it

## Known Incomplete Areas

`_on_timer` references several attributes (`self._task`, `self._latest_positions`, `self._latest_image`, `self.n_keypoints`, `self.goc_mpc`, `target_pose`) that are not initialized in `__init__`. The node is a work-in-progress; these are expected to be wired up as the task execution logic is added.

The untracked `collector.py` at the repo root is a scratch/older version; the canonical source is `ur_data_collection/collector.py`.
