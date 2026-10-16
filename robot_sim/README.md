# Robot simulation

[Back to the project](../README.md)

MuJoCo 3.3.0 simulation of the G1 (and the version 1 Jaco arms) at the LPD8. Controller geometry and robot settings
are in `configs/`.

```bash
python -m robot_sim view --demo data/robot/rims_001      # replay a demonstration
```

| File | Contents |
|---|---|
| `sim.py` | The scene, joint motors and contact-triggered pads |
| `g1.py` | The G1 with Dex3 hands: model, limits, and the 15 ↔ 28 action mapping |
| `jaco.py` | The version 1 Jaco arms and their planner |
| `retarget.py` | Fingertip path → G1 joint targets |
| `replay.py` | Replays planned targets in physics and records a demonstration |
| `common.py` | Shared helpers |

## Policy interface

| | |
|---|---|
| Image | 224 × 224 head camera |
| State | 95 values: 28 joint positions and velocities, a clock, the last 4 pads hit, the hit count, the knob angle |
| Music | The last 4 s at 24 kHz, without the robot's own part |
| Prompt | "What action should the robot take to play along with this beat.?" |
| Actions | 25 steps × 15 targets (14 arm joints + right-hand grip), expanded to 28 joints at 60 Hz; 12 steps run before the next prediction |

Without a screen: `export MUJOCO_GL=egl` (some cloud images also need
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`).
