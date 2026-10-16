# Results

[Back to the project](../README.md) · [Download](../README.md#instructions-to-replicate-it)

| Folder | Model |
|---|---|
| `three_part/` | Trained on claps, rims and shakers |
| `rims_only/` | `three_part` fine-tuned on rims |
| `rims_student/` | 1.8M-parameter student distilled from `rims_only` |
| `jaco_v1/` | Version 1 (claps on Jaco arms): results only |

Each has `config.json`, `metrics.jsonl` (training curve) and `evaluation_test.json`; the 7B models also have their
weights and test videos.

**Evaluation:** held-out takes at 65 BPM, each played after 0, 0.5, 1, 1.5 and 2 s of silence, in MuJoCo 3.3.0.
"Right part" counts runs that played the intended part; "when right" is measured on those runs only. Muted music and
other tempos are separate runs.

## Three-part model

| Part | Right part | When right | Trained chords · chord 6 |
|---|---|---|---|
| Claps | 24 / 35 | 91% of beats hit, 42 ms | 9 / 15 · 15 / 20 |
| Rims | 1 / 40 | — | 1 / 20 · 0 / 20 |
| Shakers | 23 / 30 | volume error 0.16 (mine: 0.02) | 11 / 15 · 12 / 15 |

Muted music: stayed still in 19 / 21. 67 BPM: right part in 10 / 21. 82 BPM: 0 / 21. On rims it played claps in 27
runs and turned the knob in 11.

## Rims-only model and student

| | Rims-only (7B) | Student (1.8M) |
|---|---|---|
| Right part | 31 / 40 | **40 / 40** |
| When right | 99% of beats hit, 18 ms | 99.8% of beats hit, 14 ms |
| Trained chords · chord 6 | 16 / 20 · 15 / 20 | 20 / 20 · 20 / 20 |
| Muted music: stayed still | 7 / 8 | 8 / 8 |
| 67 BPM · 82 BPM | 6 / 8 · 0 / 8 | 8 / 8 · 0 / 8 |

The rims-only model also plays rims on claps music, so it is only used for rims. In distillation the teacher played
the right part in 45 / 64 training takes; in the last round, with the student doing about 88% of the playing, 61 / 64.

<img src="../assets/robot_view.gif" width="224" alt="The rims-only model's head-camera view">

*The rims-only model on a held-out take, through its 224 px head camera.*

## Version 1: Jaco (claps, 7 held-out takes)

| | Model | My demonstrations |
|---|---|---|
| Beats hit | 99.8% | 100% |
| Timing error | 10.1 ms | 14.1 ms |
| First hit vs music start | slope 1.00: follows the music | |
| Muted music | 0 hits | |
| 67 BPM · 82 BPM | 100% hit, 55 ms late · 17% hit | |

![First hit vs silence before the music](../assets/jaco_lead_in.png)
