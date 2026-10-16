# Training and distillation

[Back to the project](../README.md)

Needs Linux, one large NVIDIA GPU (I used an H200) and about 50 GB of RAM. Download `data/` and `runs/` first
([instructions](../README.md#instructions-to-replicate-it)), then:

```bash
bash training/setup_oft.sh           # Python 3.10 environment with OpenVLA-OFT in ~/venv-oft
source ~/venv-oft/bin/activate
export MUJOCO_GL=egl
```

## Architecture

OpenVLA-OFT (Llama 2 7B with DINOv2 and SigLIP image encoders), with two added inputs:

| Input | Tokens |
|---|---|
| Head-camera image, 224 × 224 | 256, from OpenVLA's vision encoders |
| Robot state, 95 values | 1, from OFT's 2-layer projector |
| Last 4 s of music | 299, from frozen MERT-v1-95M and Audio-VLA's 3-layer projector |
| Prompt | Text tokens |

The language model fills 375 action slots (25 steps × 15 targets) in one pass, and an L1 regression head turns them
into the action chunk. Trained: LoRA adapters (rank 32), both projectors and the action head. Frozen: MERT and the 7B
weights.

## Train

```bash
python training/audio_policy.py train --output runs/three_part
python training/audio_policy.py train --output runs/rims_only --init runs/three_part --roles rims --steps 2500 --lr 1e-4 --eval-every 250
```

| | Three-part | Rims-only |
|---|---|---|
| Starts from | OpenVLA | `three_part` |
| Takes | All parts' training takes (108) | Rims training takes (40) |
| Steps · batch · learning rate | 10,000 · 16 · 5e-4 | 2,500 · 16 · 1e-4 |
| Best validation L1 (standing still) | 0.0104 (0.040) | 0.0133 (0.058) |

Each sample is a random moment of a random take, after 0–2 s of silence, over a random chord loop 1–5; 30% come from
around the first move. The learning rate drops 10× for the last 20% of steps, and the best validation checkpoint is
kept.

## Evaluate

```bash
python training/audio_policy.py evaluate runs/three_part --videos
python training/audio_policy.py evaluate runs/rims_only --videos
python training/audio_policy.py report "runs/*"
```

Closed loop on the test takes: 5 lead-ins at 65 BPM, plus 67 BPM, 82 BPM and muted music. Writes
`evaluation_test.json` (and videos) into the model's folder, running 5 simulations in parallel at about 20 GB of GPU
memory each. [Results →](../runs/README.md)

## Distil

```bash
python training/distill.py runs/rims_only --output runs/rims_student
python training/audio_policy.py evaluate runs/rims_student      # add --device cpu to run without a GPU
```

- **Student (1.8M parameters):** a CNN on a 112 px camera image, a 1-D CNN on a log-mel spectrogram of the last 4 s
  and a small network on the robot state, predicting the same 25 × 15 action chunk.
- **DAgger:** 4 rounds of 64 training takes. The robot plays the teacher's chunk with probability 0.5^round, else the
  student's; the teacher labels every re-plan, and the student retrains on everything so far (3,000 steps, batch 256).
- **Round 0** keeps only the takes where the teacher played the right part (45 / 64).
