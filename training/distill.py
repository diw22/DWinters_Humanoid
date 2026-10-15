"""Distil a trained policy (e.g. the 7B rims-only model) into a small student (1.8M parameters) with DAgger.

    python training/distill.py runs/rims_only --output runs/rims_student
    python training/audio_policy.py evaluate runs/rims_student

Round 0: the teacher plays training takes, and only the takes it got right are kept. Round r: the robot plays the
teacher's chunk with probability 0.5**r, else the student's; the teacher labels every re-plan, and the student is
retrained on everything so far.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_sim.common import read_json, write_json
from robot_sim.g1 import POLICY_DIM as ACTION_DIM
from training.audio_policy import CHUNK, PROPRIO_DIM, SR, load_episodes, load_music, load_policy, rollout

IMAGE, MELS, HOP = 112, 64, 240  # camera px; mel bands; 10 ms hop at 24 kHz


def filterbank(fft=1024, low=30., high=8000.):
    mel, hz = (lambda f: 2595*np.log10(1+f/700)), (lambda m: 700*(10**(m/2595)-1))
    edges = hz(np.linspace(mel(low), mel(high), MELS+2))[:, None]
    bins = np.fft.rfftfreq(fft, 1/SR)[None]
    rise, fall = (bins-edges[:-2])/(edges[1:-1]-edges[:-2]), (edges[2:]-bins)/(edges[2:]-edges[1:-1])
    return torch.tensor(np.maximum(0, np.minimum(rise, fall)), dtype=torch.float32)


FILTERS = filterbank()


def mel(windows):
    """(batch, samples at 24 kHz) -> (batch, MELS, frames) log-mel spectrogram."""
    x = torch.as_tensor(np.asarray(windows), dtype=torch.float32)
    power = torch.stft(x, 1024, HOP, window=torch.hann_window(1024), return_complex=True).abs()**2
    return torch.log(FILTERS@power+1e-6)


class Student(nn.Module):
    """Camera (3, 112, 112) uint8, log-mel of the last 4 s, robot state -> a normalized 25-step action chunk."""

    def __init__(self):
        super().__init__()
        def conv(i, o):
            return [nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU()]
        self.image = nn.Sequential(*conv(3, 32), *conv(32, 64), *conv(64, 128), *conv(128, 32), nn.Flatten(),
                                   nn.Linear(32*7*7, 128), nn.GELU())                  # 112 -> 7 px
        def tconv(i, o):
            return [nn.Conv1d(i, o, 5, 2, 2), nn.GroupNorm(8, o), nn.GELU()]
        self.audio = nn.Sequential(*tconv(MELS, 128), *tconv(128, 128), *tconv(128, 128), *tconv(128, 64), nn.Flatten(),
                                   nn.Linear(64*26, 256), nn.GELU())                   # 401 frames -> 26
        self.state = nn.Sequential(nn.Linear(PROPRIO_DIM, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128+256+256, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                  nn.Linear(512, CHUNK*ACTION_DIM), nn.Tanh())

    def forward(self, image, sound, state):
        x = torch.cat([self.image(image.float()/255), self.audio((sound.float()+6)/4), self.state(state)], 1)
        return self.head(x).reshape(-1, CHUNK, ACTION_DIM)


def inputs(image, state, heard):
    """The student's inputs at one re-plan, from what rollout() gives a policy."""
    return np.array(image.resize((IMAGE, IMAGE))).transpose(2, 0, 1), mel([heard])[0].numpy(), state


def play(net, x, device):
    with torch.no_grad():
        return net(*(torch.as_tensor(v[None]).to(device) for v in x))[0].cpu().numpy()


def load_student(run, device):
    """A distilled run with audio_policy.load_policy's interface (rollouts, evaluation)."""
    config = read_json(Path(run)/"config.json")
    net = Student().to(device).eval()
    net.load_state_dict(torch.load(Path(run)/"student.pt", map_location=device))
    return SimpleNamespace(act=lambda ep, image, state, heard: play(net, inputs(image, state, heard), device),
                           stats={k: np.array(v, np.float32) for k, v in config["stats"].items()},
                           lat=config["latency_s"], config=config)


def start(teacher_run, device):
    """Pool initializer: each worker process loads its own teacher (~20 GB of GPU memory)."""
    global TEACHER, EPISODES
    TEACHER = load_policy(teacher_run, device)
    EPISODES = {e.id: e for e in load_episodes(TEACHER.config["data"])}


def take(job):
    """One training take in the simulator: its outcome and the teacher-labelled samples (image, mel, state, chunk)."""
    episode, lead, beta, weights, seed, device = job
    student = Student().to(device).eval()
    student.load_state_dict(torch.load(weights, map_location=device))
    rng, samples = np.random.default_rng([*seed, 1]), []

    def act(ep, image, state, heard):
        x = inputs(image, state, heard)
        label = TEACHER.act(ep, image, state, heard)
        samples.append((x[0], x[1].astype(np.float16), x[2], label.astype(np.float16)))
        return label if rng.random() < beta else play(student, x, device)
    ep = EPISODES[episode]
    result = rollout(SimpleNamespace(act=act, stats=TEACHER.stats, lat=TEACHER.lat), ep, load_music(ep), lead, ep.bpm,
                     False, 12, seed)
    return result, samples


def fit(student, data, steps, batch, rng, device):
    """L1 to the teacher's chunks, with a random gain on the music (a shift in log-mel)."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
    student.train()
    for _ in range(steps):
        i = rng.integers(0, len(data[0]), batch)
        image, sound, state, chunk = (torch.as_tensor(d[i]).to(device) for d in data)
        sound = sound.float()+torch.empty(batch, 1, 1, device=device).uniform_(-1, 1)
        loss = (student(image, sound, state)-chunk.float()).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    student.eval()
    return float(loss)


def train(teacher_run, output, rounds, takes, steps, batch, max_lead, workers, seed, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(Path(teacher_run)/"config.json")
    roles = config.get("roles")
    episodes = [e.id for e in load_episodes(config["data"]) if e.split == "train" and (not roles or e.role in roles)]
    write_json(out/"config.json", dict(config, student=True, teacher=str(teacher_run), rounds=rounds, takes=takes,
                                       steps=steps, batch=batch, seed=seed))
    student, data = Student().to(device), None
    with torch.multiprocessing.get_context("spawn").Pool(workers, initializer=start, initargs=(teacher_run, device)) as pool:
        for r in range(rounds):
            beta = .5**r
            torch.save(student.state_dict(), out/"student.pt")
            jobs = [(episodes[rng.integers(len(episodes))], float(rng.uniform(0, max_lead)), beta, out/"student.pt",
                     [seed, r, k], device) for k in range(takes)]
            played = pool.map(take, jobs)
            kept = [s for result, samples in played if r > 0 or result["role_correct"] for s in samples]
            new = [np.stack(column) for column in zip(*kept)]
            data = new if data is None else [np.concatenate(pair) for pair in zip(data, new)]
            line = {"round": r, "beta": beta, "role_correct": float(np.mean([x["role_correct"] for x, _ in played])),
                    "kept_takes": sum(r > 0 or x["role_correct"] for x, _ in played), "samples": len(data[0]),
                    "train_l1": fit(student, data, steps, batch, rng, device)}
            torch.save(student.state_dict(), out/"student.pt")
            with (out/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(line)+"\n")
            print(json.dumps(line), flush=True)
    return {"output": str(out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("teacher_run", help="The trained policy to copy, e.g. runs/rims_only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--takes", type=int, default=64, help="Training takes played per round")
    parser.add_argument("--steps", type=int, default=3000, help="Student training steps per round")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-lead", type=float, default=2., help="Random silence before the music (s)")
    parser.add_argument("--workers", type=int, default=5, help="Simulator processes, each with its own teacher")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    print(json.dumps(train(**vars(parser.parse_args())), indent=2))
