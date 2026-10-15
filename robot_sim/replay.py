"""Physics replay of planned joint targets -> a robot demonstration: robot.mp4 (the head camera, lossless), robot.npz
(planned joints, states, actions, knob angles) and robot.json (events, hits, verdict, config)."""
from pathlib import Path
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np

from .common import read_json, write_json
from .sim import DrumSim, robot_module


def score_events(expected,actual,tolerance):
    expected=[e for e in expected if e.get("kind","hit")!="turn"]
    unused=set(range(len(actual)))
    errors=[]; missed=[]
    for event in expected:
        matches=[i for i in unused if (actual[i]["pad"],actual[i]["arm"],actual[i]["kind"])==
                 (event["pad"],event["arm"],event.get("kind","hit")) and abs(actual[i]["time_s"]-event["time_s"])<=tolerance]
        if matches:
            i=min(matches,key=lambda j:abs(actual[j]["time_s"]-event["time_s"]))
            unused.remove(i); errors.append(actual[i]["time_s"]-event["time_s"])
        else:
            missed.append(event)
    return {"matched":len(errors),"expected":len(expected),"missed":missed,
            "extra":[actual[i] for i in sorted(unused)],
            "timing_mae_s":float(np.mean(abs(np.array(errors)))) if errors else None,
            "timing_max_s":float(max(abs(np.array(errors)))) if errors else None,
            "hit_success":not missed and not unused}


def score_knob_target(target,sim):
    """A knob that should follow `target` ({knob, time_s, angle_rad, score_s}): the volume it gives (kit["knob_volume"])
    vs the volume the target gives, between score_s (the playing bars). Passes below a mean error of 0.08 of full volume
    (the claw's grip slips a few degrees)."""
    curve=sim.kit["knob_volume"]
    volume=lambda a:np.clip((np.degrees(-np.asarray(a))-curve["silent_below_deg"])/(curve["full_at_deg"]-curve["silent_below_deg"]),0,1)
    t=np.array([s["time_s"] for s in sim.grasp_samples])
    actual=np.array([s["angles_rad"][target["knob"]] for s in sim.grasp_samples])
    wanted=np.interp(t,target["time_s"],target["angle_rad"])
    low,high=target.get("score_s",(0,np.inf))  # only while the part plays: elsewhere the knob changes no sound
    error=abs(volume(actual)-volume(wanted))[(t>=low)&(t<=high)]
    return {"pad":target["knob"],"volume_error_mean":float(error.mean()),"volume_error_max":float(error.max()),
            "angle_error_max_rad":float(abs(actual-wanted).max())}


def score_knobs(expected,sim,target=None):
    """Use measured angles and contact forces; reference events are scoring only. `target`: a knob that follows a
    recorded angle (score_knob_target) instead of turn events; it may be touched and moved."""
    checks=[score_knob_target(target,sim)] if target else [];unexpected_motion=set();unexpected_contacts=set()
    turns=[e for e in expected if e.get("kind")=="turn"]
    release=sim.kit["knob_grasp"]["release_s"]
    close=sim.kit["knob_grasp"]["close_s"]
    for e in turns:
        end=e["time_s"]+e["duration_s"]
        samples=[s for s in sim.grasp_samples if e["time_s"]<=s["time_s"]<=end+1e-8]
        endpoint=min(sim.grasp_samples,key=lambda s:abs(s["time_s"]-end)) if sim.grasp_samples else None
        checks.append({"pad":e["pad"],"time_s":end,
                       "angle_error_rad":abs(endpoint["angles_rad"][e["pad"]]-e["angle_rad"]) if endpoint else None,
                       "two_finger_contact_fraction":float(np.mean([len(s["fingers"].get(e["pad"],[]))>=2 for s in samples])) if samples else 0.})
    for knob in sim.knobs:
        name=knob["id"]; relevant=[e for e in turns if e["pad"]==name]
        if target and name==target["knob"]:continue
        for sample in sim.grasp_samples:
            t=sample["time_s"]
            permitted=any(e["time_s"]-close-.1<=t<=e["time_s"]+e["duration_s"]+release+.1 for e in relevant)
            if sample["fingers"].get(name) and not permitted: unexpected_contacts.add(name)
            active=any(e["time_s"]-.15<=t<=e["time_s"]+e["duration_s"]+.2 for e in relevant)
            if not active:
                done=[e for e in relevant if e["time_s"]+e["duration_s"]<t]
                angle=max(done,key=lambda e:e["time_s"])["angle_rad"] if done else knob["initial_angle_rad"]
                if abs(sample["angles_rad"][name]-angle)>.05:unexpected_motion.add(name)
    good=not unexpected_motion and not unexpected_contacts and all(c["volume_error_mean"]<.08 if "volume_error_mean" in c else
        c["angle_error_rad"] is not None and c["angle_error_rad"]<.08 and c["two_finger_contact_fraction"]>=.8 for c in checks)
    return {"knobs_ok":bool(good),"knob_checks":checks,"unexpected_knob_motion":sorted(unexpected_motion),
            "unexpected_knob_contacts":sorted(unexpected_contacts)}


def replay(kit,events,time_s,joints,output,**extra):
    """Replays planned joint targets (time_s, joints) in physics and writes output/robot.{mp4,npz,json}; `extra` goes
    into robot.json (e.g. plan=..., episode=...). Returns the report and the robot's sound (sim.rate)."""
    import io
    from PIL import Image
    out=Path(output)
    out.mkdir(parents=True)
    states=[]; actions=[]; errors=[]
    with DrumSim(kit,duration_s=time_s[-1]) as sim:
        size=sim.render_size
        # The camera frames, JPEG-compressed like the policy's input in closed loop, then stored losslessly (RGB, qp 0).
        video=subprocess.Popen([get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
            "-s",f"{size}x{size}","-framerate",str(kit["control_hz"]),"-i","-","-c:v","libx264rgb","-qp","0",str(out/"robot.mp4")],
            stdin=subprocess.PIPE)
        for action in joints[1:]:
            states.append(sim.state())
            buffer=io.BytesIO(); Image.fromarray(sim.render()).save(buffer,"JPEG")
            video.stdin.write(np.asarray(Image.open(buffer).convert("RGB")).tobytes())
            sim.step(action)
            actions.append(sim.last_action.copy())
            errors.append(abs(sim.joint_positions()-action))
        metrics=score_events(events["events"],sim.events,kit["contact_tolerance_s"])
        metrics.update(score_knobs(events["events"],sim,events.get("knob_target")))
        max_error=np.max(errors,axis=0)
        # Fingers must stop against the knob before their commanded closing angle.
        module=robot_module(kit)
        tracking_ok=bool(np.all(max_error[module.ARM_INDICES]<.15) and np.all(max_error[module.FINGER_INDICES]<.18))
        report=dict(metrics,tracking_ok=tracking_ok,
                    max_joint_error=max_error.tolist(),bad_contacts=sim.bad_contacts,bad_contact_pairs=sim.bad_contact_pairs,
                    dataset_eligible=bool(tracking_ok and metrics["knobs_ok"] and metrics["hit_success"] and sim.bad_contacts==0),
                    frames=len(actions))
        # Each knob's angle at each state (after the previous step), from the sim's per-substep samples.
        samples=np.array([[s["angles_rad"][k["id"]] for k in sim.knobs] for s in sim.grasp_samples])
        knobs=np.vstack([[k["initial_angle_rad"] for k in sim.knobs],samples[len(samples)//len(actions)-1::len(samples)//len(actions)]])[:len(actions)]
        np.savez_compressed(out/"robot.npz",planned_time=time_s,planned_joints=joints,time=time_s[:-1],states=np.array(states),
                            actions=np.array(actions),knob_angles=knobs)
        write_json(out/"robot.json",dict(extra,events=events,hits=sim.events,report=report,knobs=[k["id"] for k in sim.knobs],kit=kit))
        sound=sim.audio[:round(time_s[-1]*sim.rate)].copy()
    video.stdin.close()
    if video.wait():
        raise RuntimeError("ffmpeg failed to write robot.mp4")
    return report,sound


def load_plan(demo):
    """A saved demonstration's (kit, events, time_s, joints), to replay or view it again."""
    info=read_json(Path(demo)/"robot.json")
    with np.load(Path(demo)/"robot.npz") as data:
        return info["kit"],info["events"],data["planned_time"],data["planned_joints"]
