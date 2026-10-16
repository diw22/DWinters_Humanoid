# Training and distillation

[Back to the project](../README.md)

`audio_policy.py` trains and evaluates the audio-conditioned OpenVLA-OFT model. `distill.py` trains the smaller student. Run all commands from the repository root.

## Setup

Training and teacher evaluation need Linux, a large NVIDIA GPU and about 50 GB of RAM. I used an H200. Training decodes every demonstration's camera frames into memory; teacher evaluation runs five simulations in parallel, each using about 20 GB of GPU memory.

Download `data/` and `runs/` as described in the [main README](../README.md#replicate-it), then set up the Python 3.10 environment:

```bash
bash training/setup_oft.sh
source ~/venv-oft/bin/activate
export MUJOCO_GL=egl
```

The setup script installs OpenVLA-OFT in `~/venv-oft`. On some cloud images, MuJoCo also needs:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## Model choice

A vision-language-action model takes an image and an instruction and predicts robot actions. OpenVLA combines two image encoders with a Llama 2 7B language model. The image is represented as tokens that the language model processes alongside the instruction.

[Audio-VLA](https://arxiv.org/abs/2511.09958) adds audio features as extra tokens. It uses gripper microphones to hear the robot's contacts, helping it judge task progress that may be difficult to see. I adapted this approach to external music, using the backing to provide both timing and the missing part. The robot never hears its own pads.

| | Audio-VLA | This project |
|---|---|---|
| Audio source | The robot's contacts, through gripper microphones | External backing music |
| Purpose | Information about task progress | Timing and part selection |
| Tasks per model | One model per task | Three parts in the initial model; rims later fine-tuned separately |

### Keeping the rhythm information

> **Challenge.** In this song, claps and rims share the same frequencies and differ in rhythm. Pooling audio over longer intervals can lose useful timing detail; AudioCLIP averages over about 0.2 seconds.
>
> **Solution.** I used MERT, pretrained on music, retaining a feature vector for roughly every 13 ms of audio. All 299 vectors from the four-second window become separate tokens, allowing the model to attend to individual beats.

The base model uses the OFT recipe to predict a chunk of 25 future actions in one pass, through a regression head. LoRA adapters are trained inside the language model while the original 7B weights remain frozen.

## Architecture

![Audio-conditioned OpenVLA-OFT architecture](../assets/musicVLA.drawio.png)

| Input | Representation |
|---|---|
| Head-camera image | 224 × 224 RGB, converted into 256 tokens by OpenVLA's vision encoders |
| Robot state | 95 values, converted into one token by OFT's two-layer projector |
| Last four seconds of music | Frozen MERT-v1-95M produces 299 feature vectors; a three-layer projector from Audio-VLA converts them into 299 tokens |
| Text | The fixed instruction: “Play along with this beat.” |

The state contains 28 joint positions, 28 joint velocities, a clock, the last four pads hit, hit count and knob angle. See [simulation inputs and actions](../robot_sim/README.md#policy-inputs-and-actions).

The token order is **vision → state → audio → instruction → action slots**. There are 375 action slots: 25 time steps × 15 targets. They are processed together in one forward pass, and the regression head produces the action chunk.

Each step contains 14 arm-joint targets and the right hand's grip. The simulator expands these to 28 joints and executes the first 12 steps at 60 Hz before asking for another prediction, every 0.2 seconds.

- **Trained:** LoRA adapters, audio and state projectors, and the action head.
- **Frozen:** MERT and the original 7B language-model weights.

## Train

```bash
python training/audio_policy.py train --output runs/three_part
python training/audio_policy.py train --output runs/rims_only --init runs/three_part --roles rims --steps 2500 --lr 1e-4 --eval-every 250
```

The three-part model was trained for 10,000 steps with batch size 16 on one H200. Each sample is a random moment from a random training take. The music has a random 0–2 second silence lead-in, and the backing uses a random chord loop from 1–5.

Thirty per cent of samples come from around the robot's first movement, where it must choose its part. Every demonstration starts from the same pose and stays still for the first bar, preventing the initial movement from giving away the part.

The best validation loss was **0.0104**, measured as L1 error on normalised actions. A policy that stands still scores 0.040.

### Why rims needed more training

The three-part model played rims in only 1 of 40 test runs. In 27 runs it played claps, and in 11 it turned the knob. A linear classifier on MERT features identified the missing part correctly on every validation take, suggesting the features contained the required information.

Two aspects of the action-prediction objective appeared to make the choice difficult:

1. **Similar initial movements.** Pad 5 is only 3.3 cm behind pad 1. The first 25 actions for claps and rims differ by 0.010, about the same as the model's ordinary prediction error. Reaching for the wrong pad therefore adds relatively little to the loss.
2. **Part information in the state.** After the first hit, the state records which pad was played. The model can continue that pattern without returning to the audio to identify the part.

Naming the part in the instruction, such as *“Play the rims part…”*, did not help: that experiment played rims in 0 of 40 runs.

I then trained on the 40 rims training takes alone for another 2,500 steps at a learning rate of `1e-4`. This retained the weights learned across all three parts but removed the need to choose between them. Rims success rose to 31 of 40 test runs.

The rims-only model also plays rims when given claps music. It is therefore used for rims, with the three-part model used for claps and shakers. Model selection is not automated.

## Evaluate

```bash
python training/audio_policy.py evaluate runs/three_part --videos
python training/audio_policy.py evaluate runs/rims_only --videos
python training/audio_policy.py report "runs/*"
```

Evaluation runs in closed loop on test takes and writes `evaluation_test.json` into the model's folder. `--videos` also saves videos of the test takes. Downloaded models can be evaluated directly without training.

Each held-out take is tested with 0, 0.5, 1, 1.5 and 2 seconds of silence before the music. Muted audio and tempo changes are separate checks. [Full results and interpretation →](../runs/README.md)

## Distil

The 7B teacher needs a large GPU to predict every 0.2 seconds, making it expensive to run and to improve further with reinforcement learning. Distillation trains a smaller model to copy it.

```bash
python training/distill.py runs/rims_only --output runs/rims_student
python training/audio_policy.py evaluate runs/rims_student
```

The student's evaluation can also run on a CPU:

```bash
python training/audio_policy.py evaluate runs/rims_student --device cpu
```

### Student architecture

The student has 1.8M parameters, about 4,000 times fewer than the teacher:

- A small CNN processes a 112 px camera image.
- A 1-D CNN processes a spectrogram of the last four seconds of music.
- A small network processes the robot state.

It predicts the same 25-step action chunk. It is trained from the rims-only teacher on rims training takes.

### DAgger

Training uses four rounds of 64 takes. In round 0, the teacher plays. In later rounds, the student performs about 50%, 75% and 88% of the playing, while the teacher labels every moment. This exposes the student to states caused by its own mistakes and provides corrections.

> **Challenge.** The teacher succeeds on only about 70% of training runs, and many failures brush the knob. Learning from all of them could pass those errors to the student.
>
> **Solution.** In round 0, keep only the 45 successful runs out of 64. Later rounds retain every correction so the student can learn to recover.

In the final round, with the student doing about 88% of the playing, 61 of 64 runs succeeded, compared with the teacher's initial 45 of 64. On the held-out rims takes, the student scored 40 of 40.

Filtering unsuccessful teacher runs is one possible explanation for the improvement, but I didn't train an unfiltered student to isolate its effect. The student still fails at 82 BPM; all training was at 65 BPM. [Teacher–student comparison →](../runs/README.md#distilled-rims-student)

