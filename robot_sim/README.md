# Robot simulation

[Back to the project](../README.md)

This folder contains the MuJoCo simulation, planning, replay and viewer for the Unitree G1 and the earlier dual-arm Kinova Jaco setup. Controller geometry and robot settings are in `configs/`.

## Watch a demonstration

With the [data downloaded](../README.md#replicate-it), run from the repository root:

```bash
pip install -r requirements.txt
python -m robot_sim view --demo data/robot/rims_001
```

This replays a stored demonstration. To run a learned model in closed loop, use the [evaluation commands](../training/README.md#evaluate).

## G1 controls

The G1 uses Dex3 hands. Its left index finger presses pads 1 and 5; its right hand grips and turns knob 4 for shakers.

| Part | Motion |
|---|---|
| Claps | Press pad 1 on beats 2 and 4 |
| Rims | Press pad 5 in a syncopated pattern, four hits a bar |
| Shakers | Turn knob 4 to fade the shaker part in and out |

The demonstrations use a fixed, extended index-finger pose for pad presses. Joint targets respect the G1's speed and torque limits. This has only been evaluated in simulation.

## Policy inputs and actions

Every 0.2 seconds, the policy receives the head-camera image, robot state, the last four seconds of backing music and the fixed text instruction.

| Input | Contents |
|---|---|
| Image | 224 × 224 RGB head-camera frame |
| Robot state | 95 values: 28 joint positions, 28 joint velocities, a clock, the last four pads hit, hit count and knob angle |
| Music | The backing track, excluding the robot's own part |
| Instruction | “Play along with this beat.” |

The 28 joints comprise seven arm joints and seven Dex3 hand joints on each side. The policy predicts 25 steps of 15 targets: 14 arm joints and the right hand's grip. These are expanded to the 28 controlled joints and passed to MuJoCo position controllers at 60 Hz.

Only the first 12 steps are executed before the policy predicts again. This gives a new prediction every 0.2 seconds while allowing each prediction to cover a longer motion.

[Model architecture →](../training/README.md#architecture)

## Rendering without a screen

For GPU evaluation:

```bash
export MUJOCO_GL=egl
```

Some cloud images also require:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Teacher evaluation runs five simulations in parallel, each using about 20 GB of GPU memory. The distilled student's evaluation can also run on a CPU. See [setup and evaluation](../training/README.md).

## Earlier Jaco setup

Version 1 used two Kinova Jaco arms for claps only. The arm had to follow the fingertip because the gripper joints were too slow. This required running at about six times the real arm's speed and adding special-case rules, so I switched to the G1 for the three-part task.

