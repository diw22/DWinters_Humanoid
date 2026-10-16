# Video-to-robot pipeline

[Back to the project](../README.md)

`camera_to_robot.py` converts each human recording into a demonstration that can be replayed in MuJoCo and used for training. Run commands from the repository root.

```bash
pip install -r requirements.txt
python pipeline/camera_to_robot.py
```

The script reads `data/human/` and writes `data/robot/`. Conversion takes roughly one or two minutes per take and doesn't need the training GPU. [Input and output files →](../data/README.md#files)

## Tracking the movement

MediaPipe finds the hand in both camera views. The cameras are calibrated with ArUco markers, allowing the index fingertip to be triangulated in 3D relative to the controller.

![Hand tracking in both camera views](../assets/tracking.gif)

*The orange ring marks the index fingertip.*

The G1's left index fingertip follows this path. For shakers, an overhead camera reads the knob angle, and the robot's right hand grips knob 4 and turns it through the same angle.

## Timing contacts

> **Challenge.** At 30 frames per second, a hit can fall anywhere within a 33 ms frame. Tracking also struggles to distinguish a fingertip hovering 5 mm above a pad from one touching it. Early demonstrations brushed pads I hadn't played.
>
> **Solution.** Use the video for movement and `played.wav`, the recording of my own part, for contact times. The pad is identified from the fingertip's position.

Each hit is pinned to the corresponding pad at the recorded time. Between hits, the fingertip stays at least 8 mm above the pads. The planned arm targets respect the G1's speed and torque limits.

## Physics check

The demonstration is replayed in MuJoCo, recording the robot's head-camera frames, joint states and actions. A pad demonstration is kept only if it reproduces the intended hits without extra touches.

The resulting dataset contains 152 usable demonstrations from 160 recordings. Compared with my recordings:

| Part | Difference after conversion |
|---|---|
| Claps | 19 ms average timing difference |
| Rims | 7 ms average timing difference |
| Shakers | 0.021 volume error |

These measure the conversion of human performances into robot demonstrations, rather than the learned policy's performance.

| Rims | Shakers |
|---|---|
| ![Rims demonstration](../assets/human_vs_robot_rims.gif) | ![Shakers demonstration](../assets/human_vs_robot_shakers.gif) |

*Left of each clip: my recording. Right: the robot's head-camera view during replay, which is the image used for training.*

Each demonstration has a comparison video with sound at `data/robot/<take>/compare.mp4`.

## Earlier Jaco version

![Human performance and Jaco replay](../assets/jaco_human_to_robot.gif)

The original claps dataset used two Kinova Jaco arms. Their grippers were too slow for the motion, so the whole arm followed the fingertip at about six times its real speed, with additional special-case rules. Switching to the G1's Dex3 hands allowed pad presses with a fixed, extended index-finger pose within the G1's speed and torque limits.

