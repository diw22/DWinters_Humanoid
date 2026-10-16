# Kraftwerk Bot: a humanoid robot that plays along to music

Dylan Winters · Humanoid Robot Learning Internship

## Demo

https://github.com/user-attachments/assets/50867109-e994-411a-a6ed-10f99850fd18

https://github.com/user-attachments/assets/76d17af8-ead6-4789-bcae-7bb45757b394

*Left: me playing. Right: the trained model playing the same music in simulation, on its own. Neither take was used in training.*

A simulated Unitree G1 listens to a drum loop with one part missing and plays it on an Akai LPD8 controller:

| Part | Control | What it plays |
|---|---|---|
| Claps | Pad 1 | Beats 2 and 4 |
| Rims | Pad 5 | A syncopated pattern, 4 hits a bar |
| Shakers | Knob 4 | Fades the shaker part in and out |

The instruction is always *“Play along with this beat.”* The idea is for the robot to work out **what to play from the missing part, and when to play from the beat**.

I recorded 160 takes of myself playing, converted the videos into robot demonstrations, and fine-tuned OpenVLA-OFT, a vision-language-action model that I extended to hear music.

## Results

Held-out takes at 65 BPM, each played five times with different delays before the music starts:

| Part | Model | Right part played | Performance when the part is right |
|---|---|---|---|
| Claps | Three-part | 23 / 35 | 92% of beats hit, 51 ms average timing error |
| Rims | Three-part | 2 / 40 | — |
| Rims | Rims-only | 33 / 40 | 98% of beats hit, 37 ms |
| Rims | Distilled student | **40 / 40** | 99.7% of beats hit, 34 ms |
| Shakers | Three-part | 24 / 30 | Volume error 0.27; my fades: 0.02 |

Rims needed further training on its own. Choosing between models is still manual.

[Full results, including muted music, unseen chords and tempo changes →](runs/README.md)

## Background

I'd been meaning to teach a robot to play music, and this internship seemed like a good opportunity. I wanted it to use music to decide what to do and when, the way a band member does.

[Audio-VLA](https://arxiv.org/abs/2511.09958) uses sounds from a robot's contacts to help it judge task progress. Here, the robot hears the backing track but never its own pads.

## From recordings to demonstrations

I first tried claps with two simulated Kinova Jaco arms. Timing was good, but the motion required about six times their real speed. I switched to a G1 with Dex3 hands, which could press the pads within its speed and torque limits, then added rims and shakers.

Two cameras tracked my fingertip for the robot to follow. For shakers, an overhead camera tracked the knob angle.

> **Challenge.** Video tracking wasn't precise enough to time contacts. Early demonstrations brushed pads I hadn't played.
>
> **Solution.** I used the audio recording of my playing to time each hit, and kept the fingertip above the pads between hits.

| Rims | Shakers |
|---|---|
| ![Rims demonstration](assets/human_vs_robot_rims.gif) | ![Shakers demonstration](assets/human_vs_robot_shakers.gif) |

*Me on the left of each clip; the robot replaying the demonstration on the right.*

Of 160 recordings, 152 passed the simulation checks. Half the test takes use a chord loop excluded from training.

[Dataset and splits →](data/README.md) · [Video-to-robot pipeline →](pipeline/README.md)

## Learning from the music

The model takes the head-camera image, robot state and last four seconds of music, then predicts its next movements. MERT, an audio model trained on music, provides the audio features.

> **Challenge.** The robot could use shortcuts. A fixed start time let it follow a clock, while different starting movements gave away which part to play.
>
> **Solution.** I added random silence before the music and made every demonstration start from the same stationary pose for the first bar.

The Jaco model followed the delayed music and stayed still when it was muted. G1 results were less consistent; see the [full checks](runs/README.md).

### Getting rims to work

On rims takes, the model usually played claps or turned the knob, despite the audio features containing enough information to distinguish the parts.

> **Challenge.** The pads are only 3.3 cm apart. Choosing the wrong one added little to the training loss, which measures movement error. Naming the part in the instruction didn't help.
>
> **Solution.** Further training on rims alone improved it from 2 / 40 to 33 / 40, though it no longer had to identify the missing part.

### Making it smaller

I trained a 1.8M-parameter student to copy the 7B rims model, starting with successful teacher demonstrations, then using the teacher to correct the student's attempts.

The student played rims in all 40 test runs and can run on a CPU. Filtering teacher failures may explain some of the improvement; I didn't train an unfiltered student to check.

[Architecture, training and distillation →](training/README.md)

## Limitations

- **Simulation only.** I haven't tested this on a real robot.
- **One song and tempo.** Trained at 65 BPM; performance drops at 67 BPM and every model fails at 82 BPM.
- **Unreliable part selection.** Changing the backing chords can change the selected part. Rims needs a separate model.
- **Imprecise shaker fades.** Volume error is 0.27, versus 0.02 for my demonstrations.
- **Small test set.** Runs repeat held-out recordings with different start delays.

## Replicate it

Download `data/` and `runs/` from Hugging Face ([data](https://huggingface.co/datasets/diw22/DWinters_Humanoid-data),
[models](https://huggingface.co/diw22/DWinters_Humanoid-models)) into the repository root:

```bash
pip install -U huggingface_hub
hf download diw22/DWinters_Humanoid-data --repo-type dataset --local-dir data     # ~11 GB
hf download diw22/DWinters_Humanoid-models --local-dir runs                       # ~3.6 GB
```

To watch a demonstration:

```bash
pip install -r requirements.txt
python -m robot_sim view --demo data/robot/rims_001
```

Training and teacher evaluation need Linux, a large NVIDIA GPU and about 50 GB of RAM; I used an H200. The viewer and video conversion pipeline don't need the training GPU.

| Folder | Details |
|---|---|
| [data/](data/README.md) | Recordings, music, splits and file formats |
| [pipeline/](pipeline/README.md) | Tracking and converting videos into demonstrations |
| [robot_sim/](robot_sim/README.md) | Simulation, robot controls and viewer |
| [training/](training/README.md) | Setup, architecture, training, evaluation and distillation commands |
| [runs/](runs/README.md) | Full results and evaluation notes |

