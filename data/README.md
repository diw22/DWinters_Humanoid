# Data

[Back to the project](../README.md)

The dataset contains recordings of me playing claps, rims and shakers, plus robot demonstrations made from those recordings. `data/` is excluded from git because of its size. See the [download instructions](../README.md#replicate-it).

## The music

The backing is one song at 65 BPM, arranged in 13-bar cycles: a kick, a chord loop with six variations, claps, rims and shakers. The robot hears the kick, chords and the two percussion parts it isn't playing. Its own part is never in the mix.

| Part | Control | Pattern |
|---|---|---|
| Claps | Pad 1 | Beats 2 and 4 |
| Rims | Pad 5 | A syncopated pattern, four hits a bar |
| Shakers | Knob 4 | Fades the shaker part in and out |

Rims and shakers takes are 11 bars: two bars of backing before playing, eight bars of playing and one bar after. The claps takes used for the G1 are about nine bars, from the original Jaco recording session.

The instruction is always *“Play along with this beat.”* It doesn't name the missing part.

## Recordings and splits

| Part | Recorded | Usable robot demonstrations | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Claps | 49 | 48 | 34 | 7 | 7 |
| Rims | 63 | 57 | 40 | 9 | 8 |
| Shakers | 48 | 47 | 34 | 7 | 6 |
| **Total** | **160** | **152** | **108** | **23** | **21** |

Claps were recorded in one 30-minute sitting, rims in one roughly 50-minute sitting, and shakers in two sittings. Two cameras filmed the performances. Demonstrations were retained after the [physics checks](../pipeline/README.md#physics-check).

### Why the split is interleaved

My timing improved during the first recording session: average error fell from about 20 ms in the first 20 takes to about 15 ms in the last 29. A chronological split would train on the earlier, less precise performances and test on the later ones.

![Timing by take, with dataset splits](../assets/jaco_timing.png)

In each block of seven consecutive takes, the fifth goes to validation and the seventh to test. This spreads all three splits across the recordings. The same rule is used for each G1 part.

### Chord variations

- Half the test takes use chord loop 6, which never appears in training.
- Each training sample uses a random backing from chord loops 1–5, so the same performance is heard over different chords.

This is intended to make the missing percussion part informative, rather than a particular chord loop. The results still show some dependence on the chords; see the [evaluation](../runs/README.md#three-part-g1-model).

### Start conditions

Every G1 demonstration begins from the same pose, held still for the first bar of music. Earlier demonstrations moved immediately, letting the model identify the part from the robot's state before it needed to listen.

During training, a random 0–2 seconds of silence precedes the music. Evaluation repeats each test take with 0, 0.5, 1, 1.5 and 2 seconds of silence. This checks whether timing follows the audio rather than a fixed episode clock.

## Files

| Path | Contents |
|---|---|
| `human/<take>/human.mp4` | First camera recording |
| `human/<take>/human_cam2.mp4` | Second camera recording |
| `human/<take>/played.wav` | The part I played; used to identify hit times |
| `human/<take>/music.wav` | The backing the robot hears |
| `human/<take>/episode.json` | Metadata, hit times and camera calibration |
| `human/<take>/knob_target.npz` | Shaker knob targets |
| `robot/<take>/robot.mp4` | Lossless head-camera video from the robot demonstration |
| `robot/<take>/robot.npz` | Joint states, actions and knob angles |
| `robot/<take>/robot.json` | Recorded hits and demonstration checks |
| `robot/<take>/compare.mp4` | Human and robot side by side, with sound |
| `music/` | Music loops used to mix the backing tracks |
| `jaco_v1/` | Original Jaco demonstrations |

## Version 1: Jaco

The first dataset was claps only: 49 takes of about six bars, converted for two simulated Kinova Jaco arms. Of these, 48 were usable: 34 train, seven validation and seven test.

Those human claps recordings were reused for the G1 version. [Jaco results →](../runs/README.md#version-1-jaco)

