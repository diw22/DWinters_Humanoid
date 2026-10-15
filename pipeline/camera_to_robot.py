"""Human take -> G1 robot demonstration (data/human/<take> -> data/robot/<take>).

    python pipeline/camera_to_robot.py [WORKERS [TAKE ...]]     # default: every take in data/human

Pad takes (claps, rims): MediaPipe tracks the hands in both camera clips (cached next to each clip), the fingertip is
triangulated, the hits come from the recorded part (played.wav), and the robot's fingertip follows the human path onto
each pad. Shakers takes: the right hand grips knob_4 and turns it through the recorded angle (knob_target.npz).
Each take is retargeted to G1 joints and replayed in physics (robot.mp4, robot.npz, robot.json); compare.mp4 shows
human | robot.
"""
from collections import Counter
from math import gcd
from multiprocessing import Pool
from pathlib import Path
import shutil
import subprocess
import sys

from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_sim import retarget as rt
from robot_sim.common import read_json

HUMAN, ROBOT = Path("data/human"), Path("data/robot")
KIT = "configs/lpd8_g1.json"
MODEL = "third_party/mediapipe/hand_landmarker.task"
LEFT, TIP = 0, 8


def hit_times(wav, rate=16000, min_interval_s=.12):
    """Hit times (s) in the recorded part (output.wav): peaks of the positive spectral flux of a log-power STFT at
    16 kHz (the 30 fps video alone would only time a hit to ~33 ms)."""
    source_rate, x = wavfile.read(wav)
    g = gcd(rate, source_rate)
    audio = resample_poly(x.astype(np.float32)/32768, rate//g, source_rate//g).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, 512)[::160]
    spec = np.log1p(np.abs(np.fft.rfft(frames*np.hanning(512), axis=-1))**2).T.astype(np.float32)
    flux = np.maximum(0, np.diff(spec, axis=1)).mean(axis=0)
    threshold = max(float(np.median(flux) + 3*np.median(abs(flux-np.median(flux)))), float(flux.max()*.12))
    peaks, _ = find_peaks(flux, height=threshold, distance=max(1, round(min_interval_s*rate/160)))
    return ((peaks*160 + 256) / rate).tolist()


def hands(video):
    """Pixel landmarks (frames, [left, right], 21, 2); NaN when missing or outside the picture (MediaPipe guesses those).
    Cached next to the clip (<clip>_hands.npz)."""
    cache = video.with_name(video.stem+"_hands.npz")
    if cache.exists():
        with np.load(cache) as data:
            return data["time_s"], data["pixels"]
    import imageio.v2 as imageio
    import mediapipe as mp
    reader = imageio.get_reader(video)
    fps, (w, h) = reader.get_meta_data()["fps"], reader.get_meta_data()["size"]
    options = mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
                                                    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    pixels = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for i, frame in enumerate(reader):
            found = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)), round(i/fps*1000))
            row = np.full((2, 21, 2), np.nan)
            for landmarks, hand in zip(found.hand_landmarks, found.handedness):
                xy = np.array([[p.x*w, p.y*h] for p in landmarks])
                xy[(xy < 5).any(1) | (xy[:, 0] > w-5) | (xy[:, 1] > h-5)] = np.nan
                row[int(hand[0].category_name == "Right")] = xy
            pixels.append(row)
    time_s, pixels = np.arange(len(pixels))/fps, np.array(pixels)
    np.savez_compressed(cache, time_s=time_s, pixels=pixels)
    return time_s, pixels


def triangulate(P, x):
    """DLT triangulation of pixels x (V, N, 2) seen by the cameras with projections P (V, 3, 4) -> controller frame
    (N, 3); NaN where fewer than two cameras see the point."""
    use = np.isfinite(x).all(-1)
    A = np.zeros((x.shape[1], 2*len(P), 4))
    for v in range(len(P)):
        u, xy = use[v, :, None], np.nan_to_num(x[v])
        A[:, 2*v] = u*(xy[:, :1]*P[v][2]-P[v][0])
        A[:, 2*v+1] = u*(xy[:, 1:]*P[v][2]-P[v][1])
    X = np.linalg.svd(A)[2][:, -1]
    X = X[:, :3]/X[:, 3:]
    X[use.sum(0) < 2] = np.nan
    return X


def fingertip(human, cameras):
    """Left index fingertip (controller frame, m) on camera 1's frame times; NaN where fewer than two cameras see it.
    `cameras`: each clip's calibration (episode.json), camera 1 first."""
    clips = list(cameras)
    t, pixels = hands(human/clips[0])
    x = [pixels[:, LEFT, TIP]]
    for name in clips[1:]:  # the other camera, resampled onto camera 1's frame times
        tv, pv = hands(human/name)
        x.append(np.stack([np.interp(t, tv, pv[:, LEFT, TIP, k], left=np.nan, right=np.nan) for k in range(2)], 1))
    return t, triangulate(np.array([c["projection"] for c in cameras.values()]), np.array(x))


def to_robot(b, kit):
    """Controller frame -> robot frame (points (..., 3))."""
    return (np.asarray(kit["board_origin_robot_m"])+b[..., :1]*kit["board_x_robot"]+b[..., 1:2]*kit["board_y_robot"]
            + b[..., 2:]*[0, 0, 1])


def robot_path(times, t, tip, hits, press, kit, bounds, hold_s):
    """Human fingertip -> left tool path (robot frame) at the control times; each hit lands on its pad at the press
    depth, where the human landed: only a landing within kit["pad_landing_margin_m"] of the pad's edge (bounds: the
    hit pads' [x0, y0, x1, y1], controller frame) is pulled in, so the robot's finger stays on the pad."""
    # Each hit on its control frame (the one the joint smoothing pins): between frames, a fast stroke's bottom fell up to
    # 4 mm above the pinned frame and the pad didn't play.
    hits = times[np.clip(np.round(np.asarray(hits)*kit["control_hz"]).astype(int), 0, len(times)-1)]
    ok = np.isfinite(tip[:, 0])
    t, tip = t[ok], tip[ok]
    # Landing = first frame within 3 mm of the lowest fingertip point within 0.1 s of a hit. Video and audio disagree by
    # up to about a frame (33 ms) and the finger moves fast there, so the camera time is warped to put each landing on
    # its hit (else the snap below uses a finger still in the air and drags the rest of the path into the pad).
    landing = {}
    for n, h in enumerate(hits):
        near = np.flatnonzero((t > h-.1) & (t < h+.1))
        if len(near):
            landing[n] = near[np.argmax(tip[near, 2] < tip[near, 2].min()+.003)]
    if landing:
        seen = sorted(landing)
        at = t[[landing[n] for n in seen]]
        t = t+np.interp(t, at, hits[seen]-at)
    span = np.clip(times, t[0], t[-1])
    board = gaussian_filter1d(np.stack([PchipInterpolator(t, tip[:, k])(span) for k in range(3)], 1), 2, axis=0)
    landed = np.array([[np.interp(h, times, board[:, k]) for k in range(2)] for h in hits])
    margin = kit["pad_landing_margin_m"]
    target = press.copy()
    target[:, :2] = to_robot(np.c_[np.clip(landed, bounds[:, :2]+margin, bounds[:, 2:]-margin), np.zeros(len(hits))], kit)[:, :2]
    xyz = to_robot(board, kit)
    at = np.array([[np.interp(h, times, xyz[:, k]) for k in range(3)] for h in hits])
    xyz += np.stack([np.interp(times, hits, (target-at)[:, k]) for k in range(3)], 1)
    # The controller is solid: over a pad the fingertip presses at most press_depth_m in (the pad gives), elsewhere it
    # stays at the pads' height (they stand above the case).
    x, y = ((xyz-kit["board_origin_robot_m"])@np.array([kit["board_x_robot"], kit["board_y_robot"]]).T).T
    on_pad = np.any([(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) for x0, y0, x1, y1 in (p["bounds_m"] for p in kit["pads"])], axis=0)
    xyz[:, 2] = np.maximum(xyz[:, 2], kit["board_origin_robot_m"][2]-np.where(on_pad, kit["press_depth_m"], 0))
    low, high = rt.tool_bounds(kit)
    xyz = np.clip(xyz, low[:3]+1e-3, high[:3]-1e-3)
    # The robot stays parked for hold_s (the episode's first bar: it hears the music before it moves, so which part it
    # plays can only come from the music), then joins the human path within kit["join_s"].
    park = rt.tool_park(kit)[:3]
    xyz = park+(xyz-park)*rt.smoothstep5(np.clip((times-hold_s)/kit["join_s"], 0, 1))[:, None]
    # Contacts come from the audio: the fingertip stays contact_clearance_m above the pads' surface except from
    # contact_lead_s before each hit until the human lifts back off (at the latest when the next hit's approach starts:
    # a pad has to be released to play again). The cameras can't tell a hover from a touch, and lose the fast downswing.
    clear, lead, surface = kit["contact_clearance_m"], kit["contact_lead_s"], kit["board_origin_robot_m"][2]
    z, floor = xyz[:, 2]-surface, np.full(len(times), clear)
    for n, h in enumerate(hits):
        a, i = np.searchsorted(times, h-lead), np.searchsorted(times, h)
        floor[a:i] = np.minimum(floor[a:i], clear*rt.smoothstep5((h-times[a:i])/lead))
        lift, end = i, np.searchsorted(times, hits[n+1]-2*lead) if n+1 < len(hits) else len(times)
        while lift < end and z[lift] < clear:
            lift += 1
        floor[i:lift] = -np.inf
    xyz[:, 2] = surface+np.maximum(z, floor)
    return xyz


def finish(human, robot, kit, events, plan, line):
    """Physics replay -> data/robot/<take>/robot.{mp4,npz,json}, compare.mp4 (human | robot; sound: the music the robot
    heard + what it played) and the verdict."""
    from robot_sim.replay import replay
    time_s, joints, planning = plan
    report, sound = replay(kit, events, time_s, joints, robot, plan=planning)
    rate, music = wavfile.read(human/"music.wav")
    played = kit["audio_sample_rate"]
    sound = resample_poly(sound.astype(np.float32), rate//gcd(rate, played), played//gcd(rate, played))
    if (human/"part.wav").exists():  # shakers: the part at the volume the robot's knob gives
        curve, (_, part) = kit["knob_volume"], wavfile.read(human/"part.wav")
        with np.load(robot/"robot.npz") as data:
            t, angle = data["time"], data["knob_angles"][:, read_json(robot/"robot.json")["knobs"].index(curve["knob"])]
        volume = np.clip((-np.degrees(angle)-curve["silent_below_deg"])/(curve["full_at_deg"]-curve["silent_below_deg"]), 0, 1)
        sound = np.r_[sound, np.zeros(len(part))][:len(part)]+part*np.interp(np.arange(len(part))/rate, t, volume)
    n = min(len(music), len(sound))
    mix = robot/"heard_and_played.wav"
    wavfile.write(mix, rate, np.clip(music[:n]+sound[:n], -32768, 32767).astype(np.int16))
    subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(human/"human.mp4"), "-i", str(robot/"robot.mp4"), "-i", str(mix),
                    "-filter_complex", "[0:v]scale=-2:480,fps=30[a];[1:v]scale=-2:480,fps=30[b];[a][b]hstack",
                    "-map", "2:a", "-shortest", "-c:v", "libx264", "-crf", "23", str(robot/"compare.mp4")], check=True)
    mix.unlink()
    if report["dataset_eligible"]:
        return line+" -> robot eligible"
    knob = next((c for c in report["knob_checks"] if "volume_error_mean" in c), None)
    return line+(f" -> robot NOT ELIGIBLE: {report['matched']}/{report['expected']} hits, {len(report['extra'])} extra, "
                 f"{report['bad_contacts']} bad contacts, tracking_ok {report['tracking_ok']}"
                 + (f", knob volume error mean {knob['volume_error_mean']:.3f} max {knob['volume_error_max']:.2f}" if knob else ""))


def knob_take(name, human, robot, kit, meta):
    """A shakers episode: the right hand grips knob_4 like the human (robot_sim.retarget.knob_claw) and turns it through
    the recorded angle (knob_target.npz); the replay scores the volume that gives in the playing bars (bars 1-8)."""
    bar = 4*60/meta["bpm"]
    with np.load(human/"knob_target.npz") as data:
        target = {"knob": kit["knob_claw"]["knob"], "time_s": data["time_s"].tolist(), "angle_rad": data["angle_rad"].tolist(),
                  "score_s": [2*bar, 10*bar]}
    times = rt.control_times(kit, meta["duration_s"])
    angle = np.interp(times, target["time_s"], target["angle_rad"])
    line = f"{name}: knob turned up to {np.degrees(-angle.min()):.0f} deg clockwise"
    shutil.rmtree(robot, ignore_errors=True)
    events = {"events": [], "knob_target": target}
    try:
        joints, path = rt.knob_claw(kit, times, angle)
        plan = rt.retarget(events, kit, meta["duration_s"], path, side="right", reference=joints)
    except ValueError as error:
        return line+f" -> RETARGET FAILED: {error}"
    return finish(human, robot, kit, events, plan, line)


def take(name):
    human, robot = HUMAN/name, ROBOT/name
    kit = read_json(KIT)
    meta = read_json(human/"episode.json")
    if meta.get("role") == "shakers":
        return knob_take(name, human, robot, kit, meta)
    reference = meta["midi_events"]
    beat = 60/meta["bpm"]
    heard = hit_times(human/"played.wav")
    if reference:  # claps: the MIDI file times the claps that made no sound
        silent = [min((r["time_s"] for r in reference), key=lambda x: abs(x-(b*beat-meta["source"]["start_s"])))
                  for b in meta["source"]["filled_beats"]]
    else:  # rims: an expected hit with no sound within 0.12 s (a touch the pad missed) plays on the grid, shifted by the player's median timing
        expected, heard_ = np.array(meta["expected_hits_s"]), np.array(heard)
        gap = np.array([heard_[np.argmin(abs(heard_-e))]-e if len(heard_) else np.inf for e in expected])
        lead = np.median(gap[abs(gap) < .12]) if np.any(abs(gap) < .12) else 0.
        silent = (expected[abs(gap) >= .12]+lead).tolist()
    hits = np.array(sorted(heard+silent))
    t, tip = fingertip(human, meta["cameras"])
    centres = {p["id"]: np.add(p["bounds_m"][:2], p["bounds_m"][2:])/2 for p in kit["pads"]}
    pads = []
    for h in hits:  # the fingertip just before and while it rests on the pad
        near = tip[(t > h-.05) & (t < h+.2)]
        near = near[np.isfinite(near[:, 0])]
        pad = min(centres, key=lambda p: np.linalg.norm(np.median(near[:, :2], 0)-centres[p])) if len(near) else None
        pads.append(pad if pad and np.linalg.norm(np.median(near[:, :2], 0)-centres[pad]) < .025 else None)
    matched = sum(any(abs(h-r["time_s"]) < .05 and p == r["pad"] for h, p in zip(hits, pads)) for r in reference)
    line = (f"{name}: {len(hits)} hits ({len(silent)} silent, from the grid), pads {dict(Counter(pads))}, "
            f"{matched}/{len(reference)} match MIDI within 50 ms; fingertip seen by both cameras {np.isfinite(tip[:, 0]).mean():.0%}")
    if None in pads:
        return line+" -> NO PAD under the fingertip at a hit; check the clip"
    events = {"events": [{"time_s": round(float(h), 4), "pad": p, "kind": "hit", "arm": "left"} for h, p in zip(hits, pads)]}
    times = rt.control_times(kit, meta["duration_s"])
    press = np.array([rt.target_xyz(rt.controls(kit)[p], kit) for p in pads])
    shutil.rmtree(robot, ignore_errors=True)
    try:
        bounds = np.array([next(p["bounds_m"] for p in kit["pads"] if p["id"] == pad) for pad in pads])
        plan = rt.retarget(events, kit, meta["duration_s"], robot_path(times, t, tip, hits, press, kit, bounds, 4*60/meta["bpm"]))
    except ValueError as error:
        return line+f" -> RETARGET FAILED: {error}"
    return finish(human, robot, kit, events, plan, line)


if __name__ == "__main__":  # python pipeline/camera_to_robot.py [WORKERS [TAKE ...]]
    names = sys.argv[2:] or sorted(p.name for p in HUMAN.iterdir())
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for line in pool.imap(take, names):
            print(line, flush=True)
