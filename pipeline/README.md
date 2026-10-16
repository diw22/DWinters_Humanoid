# Video-to-robot pipeline

[Back to the project](../README.md)

```bash
python pipeline/camera_to_robot.py      # data/human -> data/robot; 1–2 min per take, no GPU
```

1. **Track:** MediaPipe finds the hand in both cameras; the index fingertip is triangulated in 3-D using the ArUco
   calibration.
2. **Time the hits:** from `played.wav`; the pad is the one under the fingertip.
3. **Plan:** the G1's left index fingertip follows my path, with each hit pinned to its pad at its recorded time and
   the fingertip at least 8 mm above the pads in between, within the G1's speed and torque limits.
4. **Shakers:** the knob angle comes from the overhead camera (`knob_target.npz`); the right hand grips knob 4 and turns
   it to match.
5. **Replay** in MuJoCo 3.3.0, recording `robot.mp4`, `robot.npz` and `robot.json`. A demo is kept only if it
   reproduces the hits without extra contacts.

152 of 158 demos pass. Against my recordings: 19 ms (claps) and 7 ms (rims) average timing difference, 0.021 shaker
volume error.
