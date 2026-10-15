"""Train, evaluate and compare the audio-conditioned drumming policy (OpenVLA-OFT + MERT audio) on the G1 demos.

    python training/audio_policy.py train --output runs/three_part                    # all three parts
    python training/audio_policy.py train --output runs/rims_only --init runs/three_part --roles rims --steps 2500 --lr 1e-4 --eval-every 250
    python training/audio_policy.py evaluate runs/three_part --videos                 # closed loop on the test takes
    python training/audio_policy.py report "runs/*"

The policy sees the head camera, the robot state and the last 4 s of music (every part except its own) and predicts
chunks of 25 actions. Evaluation plays the test takes in the simulator: right part, hit timing / knob volume, lead-in
and tempo changes, and silence. Set up the GPU machine with training/setup_oft.sh.
"""
import argparse
import glob
import io
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_sim.common import read_json, write_json
from robot_sim.g1 import POLICY_DIM as ACTION_DIM, STATE_DIM, from_policy, to_policy

SR, WINDOW_S = 24000, 4.                   # MERT sample rate; 4 s of past music (more than a bar at 65 BPM)
CAMERA = 224                               # head-camera frames (px), robot_sim.sim.DrumSim's render size
PART_GAIN = {"claps": 1.75, "rims": 1., "shakers": 3.4}   # each part's level in the robot's music (data/music)
TRAIN_CHORDS = range(1, 6)                 # chord loops heard in training (chords6 only in half of the test takes)
CHUNK = 25                                 # actions per chunk (OFT's ALOHA setting), 60 Hz control
DECIDE_SHARE = .3                          # training samples drawn around the first move (the role decision)
HISTORY, PAD_IDS = 4, [f"pad_{i}" for i in range(1, 9)]
PROPRIO_DIM = STATE_DIM+1+HISTORY*(len(PAD_IDS)+1)+1+1   # state, clock, last pads hit, hit count, knob angle
ROLES = {"claps": "pad_1", "rims": "pad_5", "shakers": "knob_4"}   # each role's control
VLA_ID, MERT_ID, MERT_DIM = "openvla/openvla-7b", "m-a-p/MERT-v1-95M", 768


# ---------------------------------------------------------------- data

def load_episodes(root):
    """Every eligible demonstration in root/robot/<take>, with its take from root/human/<take>."""
    episodes = []
    for robot in sorted(Path(root, "robot").iterdir()):
        info = read_json(robot/"robot.json")
        if not info["report"]["dataset_eligible"]:
            continue
        human, kit = Path(root, "human", robot.name), info["kit"]
        meta = read_json(human/"episode.json")
        with np.load(robot/"robot.npz") as data:
            states, actions = data["states"].astype(np.float32), to_policy(data["actions"], kit)
            knob = data["knob_angles"][:, info["knobs"].index(ROLES["shakers"])].astype(np.float32)
        hits = sorted((e["time_s"], e["pad"]) for e in info["hits"] if e["kind"] == "hit")
        cued = meta["expected_hits_s"]
        episodes.append(SimpleNamespace(
            robot=robot, human=human, id=meta["episode_id"], split=meta["split"], session=meta["session_id"], kit=kit, role=meta["role"],
            hz=kit["control_hz"], instruction=meta["instruction"],
            bpm=meta["bpm"], go=cued[0] if cued else 2*4*60/meta["bpm"],  # first cued hit; shakers: the first playing bar
            pads=[ROLES[meta["role"]]]*len(cued), hits=hits, hit_beats=np.array(meta["hit_beats"], float),  # beats after the first
            target=info["events"].get("knob_target"),
            states=states, actions=actions, knob=knob, duration=len(actions)/kit["control_hz"]))
    splits = {}
    for e in episodes:
        if splits.setdefault(e.session, e.split) != e.split:
            raise ValueError(f"Session {e.session} appears in two splits")
    return episodes


def grid(ep, latency, lead=0., bpm=None):
    """Count-in and cued hit times (episode time), shifted by lead-in; another bpm = the take time-stretched."""
    beat, go = 60/(bpm or ep.bpm), ep.go*ep.bpm/(bpm or ep.bpm)
    count_in = [lead+go+latency-j*beat for j in range(1, int(go/beat+1e-9)+1)]
    return count_in, [lead+go+latency+k*beat for k in ep.hit_beats]


def stretch(track, factor):
    """Time-stretch (no pitch change): factor > 1 plays faster."""
    raw = subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                          "-filter:a", f"atempo={factor}", "-f", "f32le", "-"], input=track.astype(np.float32).tobytes(),
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def demo_offsets(ep):
    _, beats = grid(ep, 0.)
    times = np.array([h for h, _ in ep.hits])
    return np.array([times[np.argmin(abs(times-b))]-b for b in beats if len(times) and abs(times-b).min() < .3])


def latency(episodes):
    """One constant cue-to-hit delay (camera/display/reaction), estimated on training episodes only."""
    return float(np.median([np.median(o) for o in map(demo_offsets, episodes) if len(o)]))


def load_music(ep, name="music.wav"):
    """The music the robot hears in this take (data/human/<take>/music.wav) at SR."""
    raw = subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-i", str(ep.human/name), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


@lru_cache(None)
def loop(path):
    """One part's loop from data/music (int16 stereo, 44.1 kHz), read once."""
    from scipy.io import wavfile
    return wavfile.read(path)[1]


def backing(music, role, first, bars, chord):
    """The robot's music for a take (int16, SR): `bars` bars from project bar `first`, the last one silent, of the chord
    loop + kick + the two parts other than `role`, mixed from data/music (each part's 13-bar loop, 44.1 kHz) exactly as
    the take's music.wav was."""
    rate, bar = 44100, 4*60/65

    def part(name, source):
        audio = loop(music/name)
        return np.concatenate([audio[int(source(b)*bar*rate):int(source(b)*bar*rate)+int(round(bar*rate))].mean(1)
                               for b in range(first, first+bars-1)]).astype(np.float32)
    parts = [part("kick.wav", lambda b: b % 13), part(f"chords{chord}.wav", lambda b: (b-first) % 9)]
    for other in PART_GAIN:
        if other != role:  # the shakers loop is the 8 playing bars
            parts.append(PART_GAIN[other]*part(f"{other}.wav", (lambda b: 1+b % 8) if other == "shakers" else (lambda b: b % 13)))
    mix = np.sum([x[:min(map(len, parts))] for x in parts], axis=0)
    mix *= min(1., 32000/np.abs(mix).max())  # no clipping
    mix = np.r_[mix, np.zeros(int(round(bar*rate)), np.float32)]
    return resample_poly(mix, 160, 294).clip(-32768, 32767).astype(np.int16)  # 44.1 -> 24 kHz


def load_variants(ep):
    """Training: the take's performance over every trained chord loop (same parts, other chords), so only the missing
    part tells the role."""
    meta = read_json(ep.human/"episode.json")
    return [backing(ep.human.parents[1]/"music", ep.role, meta["first_bar"], meta["bars"], k).astype(np.float32)/32768
            for k in TRAIN_CHORDS]


def camera_frames(ep):
    """All of a demo's head-camera frames (steps, CAMERA, CAMERA, 3) uint8, from its lossless robot.mp4."""
    raw = subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-i", str(ep.robot/"robot.mp4"), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, CAMERA, CAMERA, 3)


def window(track, t, rng, noise=None):
    """The WINDOW_S seconds of music before episode time t (strictly causal), with level/noise augmentation.
    Noise covers the whole window, so the start of the take is not audible."""
    end = int(round(t*SR))
    start = end-int(WINDOW_S*SR)
    x = np.zeros(end-start, np.float32)
    lo, hi = max(start, 0), min(end, len(track))
    if hi > lo:
        x[lo-start:hi-start] = track[lo:hi]
    # Noise >= 10 dB under the quietest part (the shakers stem, RMS 0.0064): which part is missing is the role cue.
    gain, level = (rng.uniform(.5, 1.5), rng.uniform(.0003, .002)) if noise is None else (1., noise)
    return (x*gain+rng.normal(0, level, len(x))).astype(np.float32)


def proprio(ep, state, hits, clock, knob, stats):
    history = np.zeros((HISTORY, len(PAD_IDS)+1), np.float32)
    history[:, -1] = 1
    for k, (_, pad) in enumerate(reversed(hits[-HISTORY:])):
        history[k] = 0
        history[k, PAD_IDS.index(pad)] = 1
    return np.r_[(state-stats["state_mean"])/stats["state_std"], clock/20, history.ravel(), len(hits)/8, knob].astype(np.float32)


# ---------------------------------------------------------------- model

class Mert:
    """Frozen MERT: (batch, samples at 24 kHz) -> (batch, frames, 768), the average of its layers."""

    def __init__(self, device):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.device = device
        self.model = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True).to(device).eval()
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, trust_remote_code=True)
        self.tokens = self(np.zeros((1, int(WINDOW_S*SR)), np.float32)).shape[1]

    @torch.no_grad()
    def __call__(self, windows):
        x = self.extractor(list(windows), sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        return torch.stack(self.model(x, output_hidden_states=True).hidden_states).mean(0)


class Projector(nn.Module):
    """Robot-state token (OFT's 2-layer projector) plus audio tokens (Audio-VLA's 3-layer projector)."""

    def __init__(self, llm_dim, audio_tokens):
        super().__init__()
        from prismatic.models.projectors import ProprioProjector
        self.state, self.audio_tokens = ProprioProjector(llm_dim, PROPRIO_DIM), audio_tokens
        self.audio = nn.Sequential(nn.Linear(MERT_DIM, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim), nn.GELU(),
                                   nn.Linear(llm_dim, llm_dim))

    def forward(self, x):  # (batch, PROPRIO_DIM + audio_tokens*MERT_DIM)
        audio = self.audio(x[:, PROPRIO_DIM:].reshape(len(x), self.audio_tokens, MERT_DIM))
        return torch.cat([self.state(x[:, :PROPRIO_DIM])[:, None], audio], 1)


def append_tokens(self, patches, extra, projector):
    """Replaces OFT's single proprio token with all of the projector's tokens."""
    if projector is None or extra is None:
        return patches
    return torch.cat((patches, projector(extra.reshape(patches.shape[0], -1))), dim=1)


def load_vla(device, audio_tokens, lora=None, trainable=False):
    """OpenVLA-OFT with this robot's action/state sizes and the extra tokens."""
    import prismatic.vla.constants as constants
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    sizes = {"NUM_ACTIONS_CHUNK": CHUNK, "ACTION_DIM": ACTION_DIM, "PROPRIO_DIM": PROPRIO_DIM,
             "ACTION_PROPRIO_NORMALIZATION_TYPE": constants.NormalizationType.BOUNDS}
    for name, module in list(sys.modules.items()):  # OFT reads these module constants at call time
        if name.startswith("prismatic"):
            for key, value in sizes.items():
                if hasattr(module, key):
                    setattr(module, key, value)
    OpenVLAForActionPrediction._process_proprio_features = append_tokens
    # As OFT's finetune.py: use the local prismatic classes for the checkpoint's custom code.
    from transformers import AutoConfig, AutoImageProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    path = snapshot_download(VLA_ID)
    processor = PrismaticProcessor.from_pretrained(path)
    vla = OpenVLAForActionPrediction.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    vla.vision_backbone.set_num_images_in_input(1)
    llm_dim = vla.llm_dim
    if lora:
        vla = PeftModel.from_pretrained(vla, lora, is_trainable=trainable)
    else:
        vla = get_peft_model(vla, LoraConfig(r=32, lora_alpha=16, lora_dropout=0., target_modules="all-linear",
                                             init_lora_weights="gaussian"))
    projector = Projector(llm_dim, audio_tokens).to(device)
    head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM).to(device)
    return vla, processor, projector, head


# ---------------------------------------------------------------- train

class Batches:
    """Training samples with a random lead-in: the take is preceded by 0..max_lead s of waiting."""

    def __init__(self, episodes, tracks, processor, stats, max_lead):
        from prismatic.vla.action_tokenizer import ActionTokenizer
        self.episodes, self.tracks, self.processor, self.stats, self.max_lead = episodes, tracks, processor, stats, max_lead
        self.tokens = ActionTokenizer(processor.tokenizer)
        self.frames = {}  # each episode's camera frames, decoded on first use (~0.4 GB per take)

    def draw(self, rng, n):
        from PIL import Image
        items, windows = [], []
        for _ in range(n):
            e = rng.integers(len(self.episodes))
            ep = self.episodes[e]
            lead = rng.integers(0, int(self.max_lead*ep.hz)+1)
            if rng.random() < DECIDE_SHARE:  # around the robot's first move after the first bar: which part to play
                bar = int(4*60/ep.bpm*ep.hz)
                i = lead+rng.integers(bar-ep.hz//2, bar+3*ep.hz)
            else:
                i = rng.integers(0, lead+len(ep.actions))  # step in the augmented episode
            j, t = i-lead, (i-lead)/ep.hz              # step and time in the original take
            chunk = ep.actions[np.clip(np.arange(j, j+CHUNK), 0, len(ep.actions)-1)]
            if e not in self.frames:
                self.frames[e] = camera_frames(ep)
            image = Image.fromarray(self.frames[e][max(j, 0)])
            items.append(item(self.processor, self.tokens, ep.instruction, normalize(chunk, self.stats), image,
                              proprio(ep, ep.states[max(j, 0)], [h for h in ep.hits if h[0] <= t+1e-9], i/ep.hz, ep.knob[max(j, 0)],
                                      self.stats)))
            windows.append(window(self.tracks[e][rng.integers(len(self.tracks[e]))], t, rng))
        return collate(items, self.processor.tokenizer.pad_token_id), windows


def still_l1(episodes, stats, max_lead, rng, n=4000):
    """L1 of a policy that just holds its current command, on the same sample distribution: the bar to clear."""
    total = 0.
    for _ in range(n):
        ep = episodes[rng.integers(len(episodes))]
        lead = rng.integers(0, int(max_lead*ep.hz)+1)
        j = rng.integers(0, lead+len(ep.actions))-lead
        chunk = ep.actions[np.clip(np.arange(j, j+CHUNK), 0, len(ep.actions)-1)]
        total += np.abs(normalize(chunk, stats)-normalize(ep.actions[[max(j-1, 0)]], stats)).mean()
    return total/n


def normalize(actions, stats):
    return np.clip(2*(actions-stats["low"])/(stats["high"]-stats["low"]+1e-8)-1, -1, 1).astype(np.float32)


def unnormalize(actions, stats):
    return stats["low"]+(np.asarray(actions)+1)/2*(stats["high"]-stats["low"])


def item(processor, tokens, instruction, chunk, image, state):
    """One model input. The action-token slots hold `chunk`; OFT zeroes their input embeddings, so at
    inference any placeholder chunk gives the same prediction."""
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.constants import IGNORE_INDEX
    actions = "".join(tokens(chunk))
    prompt = PurePromptBuilder("openvla")
    prompt.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
    prompt.add_turn("gpt", actions)
    ids = torch.tensor(processor.tokenizer(prompt.get_prompt(), add_special_tokens=True).input_ids)
    labels = ids.clone()
    labels[:-(len(actions)+1)] = IGNORE_INDEX
    return ids, labels, processor.image_processor.apply_transform(image), chunk, state


def collate(items, pad):
    from prismatic.vla.constants import IGNORE_INDEX
    length = max(len(ids) for ids, *_ in items)
    ids = torch.stack([nn.functional.pad(i, (0, length-len(i)), value=pad) for i, *_ in items])
    labels = torch.stack([nn.functional.pad(l, (0, length-len(l)), value=IGNORE_INDEX) for _, l, *_ in items])
    return {"input_ids": ids, "attention_mask": ids != pad, "labels": labels,
            "pixel_values": torch.stack([p for _, _, p, *_ in items]),
            "actions": torch.tensor(np.stack([c for *_, c, _ in items])), "proprio": torch.tensor(np.stack([s for *_, s in items]))}


def forward(vla, projector, head, batch, audio, num_patches, device):
    """Mean L1 between predicted and demonstrated action chunks (normalized units)."""
    return nn.functional.l1_loss(predict(vla, projector, head, batch, audio, num_patches, device), batch["actions"].to(device))


def predict(vla, projector, head, batch, audio, num_patches, device):
    """Normalized action chunks (batch, CHUNK, ACTION_DIM) from one forward pass; audio = MERT features."""
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    extra = torch.cat([batch["proprio"].to(device), audio.flatten(1)], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                     pixel_values=batch["pixel_values"].to(device, torch.bfloat16), labels=batch["labels"].to(device),
                     output_hidden_states=True, proprio=extra, proprio_projector=projector)
        targets = batch["labels"][:, 1:].to(device)
        mask = get_current_action_mask(targets) | get_next_actions_mask(targets)
        hidden = output.hidden_states[-1][:, num_patches:-1][mask].reshape(len(targets), CHUNK*ACTION_DIM, -1)
        return head.predict_action(hidden.to(torch.bfloat16)).float()


def train(output, seed, data, steps, batch, max_lead, lr, eval_every, device, init=None, roles=None):
    """init: continue from a trained run (its LoRA, projector, head and normalization). roles: train (and later
    evaluate) on these parts' episodes only."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    episodes = [e for e in load_episodes(data) if not roles or e.role in roles]
    lat = latency([e for e in episodes if e.split == "train"])
    split = {s: [e for e in episodes if e.split == s] for s in ("train", "val", "test")}
    states, actions = np.concatenate([e.states for e in split["train"]]), np.concatenate([e.actions for e in split["train"]])
    stats = {"state_mean": states.mean(0), "state_std": states.std(0)+1e-3,
             "low": actions.min(0), "high": actions.max(0)}  # min/max like OFT's ALOHA: percentile clipping kept the finger off some pads
    if init:  # the trained head's outputs only mean something on its own scale
        stats = {k: np.array(v, np.float32) for k, v in read_json(Path(init)/"config.json")["stats"].items()}
    mert = Mert(device)
    tokens = mert.tokens
    vla, processor, projector, head = load_vla(device, tokens, lora=Path(init)/"lora" if init else None, trainable=True)
    if init:
        weights = torch.load(Path(init)/"heads.pt", map_location=device)
        projector.load_state_dict(weights["projector"])
        head.load_state_dict(weights["head"])
    num_patches = vla.vision_backbone.get_num_patches()+1+tokens
    tracks = {s: [load_variants(e) if s != "test" else [load_music(e)] for e in v] for s, v in split.items()}  # val as train
    train_data = Batches(split["train"], tracks["train"], processor, stats, max_lead)
    val_data = Batches(split["val"], tracks["val"], processor, stats, max_lead)
    val_rng = np.random.default_rng(seed+10_000)
    val_batches = [val_data.draw(val_rng, batch) for _ in range(max(1, 64//batch))]
    parameters = [p for p in vla.parameters() if p.requires_grad]+list(projector.parameters())+list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.8*steps)], .1)
    still = still_l1(split["val"], stats, max_lead, np.random.default_rng(seed+20_000))
    config = {"seed": seed, "max_lead_s": max_lead, "latency_s": lat, "audio_tokens": tokens, "still_val_l1": still,
              "init": str(init) if init else None, "roles": roles,
              "episodes": {s: [e.id for e in v] for s, v in split.items()}, "data": str(data),
              "stats": {k: np.asarray(v, np.float32).tolist() for k, v in stats.items()}}
    write_json(out/"config.json", config)
    print(f"{len(split['train'])} train / {len(split['val'])} val episodes, "
          f"latency {lat*1000:.0f} ms, {tokens} audio tokens, "
          f"{sum(p.numel() for p in parameters):,} trainable parameters; val L1 of standing still {still:.4f}", flush=True)

    def save(folder):  # LoRA + projector + action head (no optimizer state)
        vla.save_pretrained(folder/"lora")
        torch.save({"projector": projector.state_dict(), "head": head.state_dict(), "step": step}, folder/"heads.pt")

    best, total = float("inf"), 0.
    vla.train()
    for step in range(1, steps+1):
        inputs, windows = train_data.draw(rng, batch)
        loss = forward(vla, projector, head, inputs, mert(windows), num_patches, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        schedule.step()
        total += float(loss)
        if step % eval_every == 0 or step == steps:
            vla.eval(); projector.eval(); head.eval()
            with torch.no_grad():
                val = float(np.mean([float(forward(vla, projector, head, b, mert(w), num_patches, device))
                                     for b, w in val_batches]))
            vla.train(); projector.train(); head.train()
            line = {"step": step, "train_l1": total/eval_every, "val_l1": val, "still_l1": still}
            total = 0.
            with (out/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(line)+"\n")
            print(json.dumps(line), flush=True)
            if val < best:  # early stopping on held-out validation sessions
                best = val
                save(out)
    return {"output": str(out), "best_val_l1": best}


# ---------------------------------------------------------------- closed-loop evaluation

def score(hits, beats, pads, beat, cap=.25):
    """Each cued beat is matched to the nearest unused hit on the right pad within half a beat;
    a miss counts as `cap` seconds of error."""
    used, errors, signed = set(), [], []
    for time, pad in zip(beats, pads):
        options = [i for i, (h, p) in enumerate(hits) if i not in used and p == pad and abs(h-time) < beat/2]
        if options:
            i = min(options, key=lambda i: abs(hits[i][0]-time))
            used.add(i)
            signed.append(hits[i][0]-time)
            errors.append(min(abs(signed[-1]), cap))
        else:
            errors.append(cap)
    intervals = np.diff(sorted(h for h, _ in hits))
    return {"beat_error_ms": 1000*float(np.mean(errors)), "hit_rate": len(signed)/len(beats),
            "extra_hits": len(hits)-len(signed), "asynchrony_ms": 1000*float(np.median(signed)) if signed else None,
            "first_hit_s": float(min(h for h, _ in hits)) if hits else None,
            "tempo_ratio": float(np.median(intervals)/beat) if len(intervals) else None}


def knob_score(ep, times, angles, lead=0., scale=1.):
    """Shakers: the mean error of the volume the knob gives (kit["knob_volume"]) against the demonstration's target,
    over the playing bars; the target moved by the lead-in and stretched by a tempo change. First turn: > 5 deg."""
    curve, target = ep.kit["knob_volume"], ep.target
    volume = lambda a: np.clip((np.degrees(-np.asarray(a))-curve["silent_below_deg"])/(curve["full_at_deg"]-curve["silent_below_deg"]), 0, 1)
    times, angles = np.asarray(times), np.asarray(angles)
    wanted = np.interp((times-lead)/scale, target["time_s"], target["angle_rad"])
    low, high = lead+np.array(target["score_s"])*scale
    playing = (times >= low) & (times <= high)
    turned = np.flatnonzero(np.degrees(-angles) > 5)
    return {"volume_error": float(abs(volume(angles)-volume(wanted))[playing].mean()),
            "first_turn_s": float(times[turned[0]]) if len(turned) else None}


def played(hits, angles):
    """Which parts the robot played: >= 3 hits on a role's pad, or the knob turned > 10 deg clockwise."""
    pads = [p for _, p in hits]
    return {role: bool(np.degrees(-np.min(angles, initial=0)) > 10) if control.startswith("knob") else pads.count(control) >= 3
            for role, control in ROLES.items()}


def outcome(ep, hits, times, angles, beats=None, beat=None, lead=0., scale=1.):
    """Role accuracy (its own part and no other) + the role's score: hit timing (pads) or volume error (knob)."""
    parts = played(hits, angles)
    row = {"role": ep.role, "played": [r for r, v in parts.items() if v],
           "role_correct": bool(parts[ep.role] and sum(parts.values()) == 1)}
    if ep.role == "shakers":
        row.update(knob_score(ep, times, angles, lead, scale), extra_hits=len(hits))
        row["first_action_s"] = row["first_turn_s"]
    else:
        row.update(score(hits, beats, ep.pads, beat))
        row["first_action_s"] = row["first_hit_s"]
    return row


def demonstration(ep, lat):
    """The demonstration's own score, as a reference row."""
    return dict(outcome(ep, ep.hits, np.arange(len(ep.knob))/ep.hz, ep.knob, grid(ep, lat)[1], 60/ep.bpm), episode=ep.id)


def rollout(policy, ep, track, lead, bpm, silent, execute, seed, video=None):
    """One closed-loop take in the simulator; the policy re-predicts every `execute` steps."""
    from PIL import Image
    from robot_sim.sim import DrumSim
    p, rng, beat, scale = policy, np.random.default_rng(seed), 60/bpm, ep.bpm/bpm
    count_in, beats = grid(ep, p.lat, lead, bpm)
    if bpm != ep.bpm:
        track = stretch(track, bpm/ep.bpm)  # unseen tempo: the same take, time-stretched
    if silent:
        track = np.zeros_like(track)
    duration = lead+ep.duration*scale
    frames, times, angles = [], [], []
    with DrumSim(ep.kit, duration_s=duration) as sim:
        knob = [k["id"] for k in sim.knobs].index(ROLES["shakers"])
        for i in range(int(duration*ep.hz)):
            state = proprio(ep, sim.state(), sorted((e["time_s"], e["pad"]) for e in sim.events if e["kind"] == "hit"),
                            i/ep.hz, sim.knob_angle(knob), p.stats)
            if i % execute == 0:
                buffer = io.BytesIO()
                Image.fromarray(sim.render()).save(buffer, "JPEG")  # same compression as the training frames
                # the music, shifted by the lead-in, heard up to now
                chunk = p.act(ep, Image.open(buffer).convert("RGB"), state, window(track, i/ep.hz-lead, rng, noise=.001))
            action = chunk[i % execute]
            if video is not None:
                frames.append(sim.render())
            sim.step(from_policy(unnormalize(action, p.stats), ep.kit))
            times.append((i+1)/ep.hz)
            angles.append(sim.knob_angle(knob))
        hits = sorted((e["time_s"], e["pad"]) for e in sim.events if e["kind"] == "hit")
        if video is not None:
            heard = np.zeros(int(duration*SR), np.float32)
            heard[int(lead*SR):] = np.pad(track, (0, len(heard)))[:len(heard)-int(lead*SR)]
            robot = np.interp(np.arange(len(heard))/SR, np.arange(len(sim.audio))/sim.rate, sim.audio)
            write_video(video, frames, ep.hz, heard+robot.astype(np.float32))
    return dict(outcome(ep, hits, times, angles, beats, beat, lead, scale), episode=ep.id, lead_s=lead, bpm=bpm, silent=silent,
                bad_contacts=sim.bad_contacts)


def load_policy(run, device):
    """A trained run (its best-validation weights, or a distilled student) ready for rollouts."""
    run = Path(run)
    if (run/"student.pt").exists():
        from training.distill import load_student
        return load_student(run, device)
    from prismatic.vla.action_tokenizer import ActionTokenizer
    config = read_json(run/"config.json")
    stats = {k: np.array(v, np.float32) for k, v in config["stats"].items()}
    mert = Mert(device)
    vla, processor, projector, head = load_vla(device, config["audio_tokens"], lora=run/"lora")
    weights = torch.load(run/"heads.pt", map_location=device)
    projector.load_state_dict(weights["projector"])
    head.load_state_dict(weights["head"])
    vla.eval(); projector.eval(); head.eval()
    tokens, pad = ActionTokenizer(processor.tokenizer), processor.tokenizer.pad_token_id
    num_patches = vla.vision_backbone.get_num_patches()+1+config["audio_tokens"]
    placeholder = np.zeros((CHUNK, ACTION_DIM), np.float32)

    def act(ep, image, state, heard):
        """One normalized action chunk from the camera image, the robot state and the music heard so far."""
        with torch.no_grad():
            batch = collate([item(processor, tokens, ep.instruction, placeholder, image, state)], pad)
            return predict(vla, projector, head, batch, mert([heard]), num_patches, device)[0].cpu().numpy()
    return SimpleNamespace(act=act, stats=stats, lat=config["latency_s"], config=config)


def write_video(path, frames, fps, audio):
    import imageio.v2 as imageio
    from scipy.io import wavfile
    path = Path(path)
    imageio.mimwrite(path.with_suffix(".silent.mp4"), frames, fps=fps)
    wavfile.write(path.with_suffix(".wav"), SR, (np.clip(audio, -1, 1)*32767).astype(np.int16))
    subprocess.run([get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path.with_suffix(".silent.mp4")),
                    "-i", str(path.with_suffix(".wav")), "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
    path.with_suffix(".silent.mp4").unlink()
    path.with_suffix(".wav").unlink()


def evaluate_share(run, split, lead_ins, tempos, execute, episodes_limit, videos, device, share, workers):
    """One evaluation process's share (every `workers`-th episode of the run's parts): its rollouts, demonstration rows
    and the run config."""
    policy = load_policy(run, device)
    out = Path(run)
    roles = policy.config.get("roles") or ROLES
    episodes = [e for e in load_episodes(policy.config["data"]) if e.split == split and e.role in roles][:episodes_limit]
    rollouts, demos = [], []
    for n, ep in enumerate(episodes):
        if n % workers != share:
            continue
        track = load_music(ep)
        demos.append(demonstration(ep, policy.lat))
        conditions = [("lead", d, ep.bpm, False) for d in lead_ins]+[("tempo", 1., b, False) for b in tempos]
        conditions += [("silent", 1., ep.bpm, True)]
        for c, (kind, lead, bpm, silent) in enumerate(conditions):
            video = out/"videos"/f"{ep.id}_{kind}_{lead:g}s_{bpm:g}bpm.mp4" if videos and (kind != "lead" or lead == 1.) else None
            result = rollout(policy, ep, track, lead, bpm, silent, execute, seed=[99, n, c], video=video)
            rollouts.append(dict(result, condition=kind))
            print(json.dumps(rollouts[-1]), flush=True)
    return rollouts, demos, policy.config


def evaluate(run, split, lead_ins, tempos, execute, episodes_limit, videos, device, workers=5):
    """Closed-loop evaluation, split over `workers` processes on the one GPU (each loads its own copy of the policy,
    ~20 GB); same seeds as one process, so the same results."""
    out = Path(run)
    if videos:
        (out/"videos").mkdir(exist_ok=True)
    shares = [(run, split, lead_ins, tempos, execute, episodes_limit, videos, device, k, workers)
              for k in range(workers)]
    if workers == 1:
        parts = [evaluate_share(*shares[0])]
    else:
        with torch.multiprocessing.get_context("spawn").Pool(workers) as pool:
            parts = pool.starmap(evaluate_share, shares)
    rollouts, demos, config = [r for p in parts for r in p[0]], [d for p in parts for d in p[1]], parts[0][2]
    summary = summarize(rollouts, demos)
    write_json(out/f"evaluation_{split}.json", {"config": {k: v for k, v in config.items() if k != "stats"},
                                                "summary": summary, "rollouts": rollouts, "demonstrations": demos})
    return summary


def summarize(rollouts, demos):
    def mean(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    def scores(rows, prefix=""):
        pads = [r for r in rows if r["role"] != "shakers"]
        out = {f"{prefix}role_accuracy": mean(rows, "role_correct"), f"{prefix}beat_error_ms": mean(pads, "beat_error_ms"),
               f"{prefix}hit_rate": mean(pads, "hit_rate"),
               f"{prefix}volume_error": mean([r for r in rows if r["role"] == "shakers"], "volume_error")}
        for role in ROLES:
            out[f"{prefix}{role}_role_accuracy"] = mean([r for r in rows if r["role"] == role], "role_correct")
        return out
    lead = [r for r in rollouts if r["condition"] == "lead"]
    summary = dict(scores(lead), extra_hits=mean(lead, "extra_hits"))
    started = [r for r in lead if r["first_action_s"] is not None]
    if len({r["lead_s"] for r in started}) > 1:  # each episode's own first-action time removed: only the lead-in's effect is left
        x, y = np.array([r["lead_s"] for r in started]), np.array([r["first_action_s"] for r in started])
        for episode in {r["episode"] for r in started}:
            mine = np.array([r["episode"] == episode for r in started])
            x[mine], y[mine] = x[mine]-x[mine].mean(), y[mine]-y[mine].mean()
        slope = float(np.polyfit(x, y, 1)[0])
        summary.update(lead_in_slope=slope, lead_in_r2=float(1-np.sum((y-slope*x)**2)/max(np.sum(y**2), 1e-12)))
    for kind in ("tempo", "silent"):
        rows = [r for r in rollouts if r["condition"] == kind]
        if rows:
            summary.update(scores(rows, f"{kind}_"))
            summary[f"{kind}_played_anything"] = float(np.mean([bool(r["played"]) for r in rows]))
    summary.update({f"demonstration_{k}": v for k, v in scores(demos).items()})
    return summary


# ---------------------------------------------------------------- report

def report(runs, split):
    groups = {}
    for run in runs:
        path = Path(run)/f"evaluation_{split}.json"
        if path.exists():
            result = read_json(path)
            groups.setdefault(Path(run).name, []).append(result)
    if not groups:
        raise ValueError(f"No evaluation_{split}.json in the given runs")
    keys = ["role_accuracy", "claps_role_accuracy", "rims_role_accuracy", "shakers_role_accuracy", "beat_error_ms", "hit_rate",
            "volume_error", "extra_hits", "lead_in_slope", "tempo_role_accuracy", "tempo_beat_error_ms", "tempo_volume_error",
            "silent_played_anything", "demonstration_beat_error_ms", "demonstration_volume_error"]
    table = {}
    for name, results in groups.items():
        table[name] = {"runs": len(results)}
        for key in keys:
            values = [r["summary"].get(key) for r in results if r["summary"].get(key) is not None]
            if values:
                table[name][key] = f"{np.mean(values):.3g} ± {np.std(values):.2g}" if len(values) > 1 else f"{values[0]:.3g}"
    print(json.dumps(table, indent=2))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data", help="data/human/<take> + data/robot/<take>")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-lead", type=float, default=2., help="Random lead-in range (s); 0 disables the augmentation")
    p.add_argument("--lr", type=float, default=5e-4, help="OFT's LoRA learning rate; x0.1 for the last 20%% of steps")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--init", help="Continue from this trained run (its weights and normalization)")
    p.add_argument("--roles", nargs="+", choices=list(ROLES), help="Train on these parts' takes only (e.g. rims); evaluation follows")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("evaluate")
    p.add_argument("run")
    p.add_argument("--split", default="test")
    p.add_argument("--lead-ins", type=float, nargs="+", default=[0., .5, 1., 1.5, 2.])
    p.add_argument("--tempos", type=float, nargs="*", default=[67., 82.], help="Unseen tempos (BPM)")
    p.add_argument("--execute", type=int, default=12, help="Actions executed from each 25-action chunk before re-predicting")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--videos", action="store_true")
    p.add_argument("--workers", type=int, default=5, help="Evaluation processes on the one GPU (~20 GB each)")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+")
    p.add_argument("--split", default="test")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "train":
        print(json.dumps(train(**args), indent=2))
    elif command == "evaluate":
        args["episodes_limit"] = args.pop("episodes")
        print(json.dumps(evaluate(**args), indent=2))
    else:
        report([r for pattern in args["runs"] for r in glob.glob(pattern)], args["split"])
