# Data

[Back to the project](../README.md) · [Download](../README.md#instructions-to-replicate-it)

Recordings of me playing three parts at 65 BPM, and the G1 demonstrations made from them.

| Part | Control | Recorded | Usable demos | Train | Val | Test |
|---|---|---:|---:|---:|---:|---:|
| Claps | Pad 1 | 49 | 48 | 34 | 7 | 7 |
| Rims | Pad 5 | 63 | 57 | 40 | 9 | 8 |
| Shakers | Knob 4 | 48 | 47 | 34 | 7 | 6 |

- **Split:** in each block of 7 consecutive takes, the 5th goes to validation and the 7th to test.
- **Music:** the robot hears the kick, the chords and the two parts it isn't playing. Training uses chord loops 1–5 at
  random; half the test takes use chord loop 6, which training never uses.
- **Not used:** rims_016 and shakers_043 have no demo; 6 demos fail the physics check (`report.dataset_eligible` in
  `robot.json`) and training skips them.

| Path | Contents |
|---|---|
| `human/<take>/human.mp4`, `human_cam2.mp4` | My recording from two cameras (shakers takes: one) |
| `human/<take>/played.wav` | The part I played, used for the hit times |
| `human/<take>/music.wav` | What the robot hears (24 kHz) |
| `human/<take>/episode.json` | Part, split, hit times, camera calibration |
| `human/<take>/*_hands.npz` | Cached hand tracking |
| `human/<take>/knob_target.npz`, `part.wav` | Shakers only: the knob-angle target and the shaker part's audio |
| `robot/<take>/robot.mp4` | Head-camera video, lossless: the policy's image input |
| `robot/<take>/robot.npz` | Joint states, actions, knob angles |
| `robot/<take>/robot.json` | Hits, checks, controller layout |
| `robot/<take>/compare.mp4` | Me and the robot side by side, with sound |
| `music/` | The song's loops (44.1 kHz); training mixes the chord variations from them |
| `jaco_v1/<take>/` | Version 1: 49 claps demos on two Jaco arms |
