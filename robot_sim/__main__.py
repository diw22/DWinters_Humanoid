"""Watch the scene, or loop a saved trajectory, in the MuJoCo viewer (from the repository root):

    python -m robot_sim view [--demo data/robot/rims_001] [--speed 0.5]
"""
import argparse
import time

import mujoco.viewer

from .common import read_json
from .replay import load_plan
from .sim import DrumSim


def view(demo=None,speed=1.):
    targets=None
    if demo:
        kit,_,_,targets=load_plan(demo)
    else:
        kit=read_json("configs/lpd8_g1.json")
    state={"paused":False}
    def key(code):
        if code==32: state["paused"]=not state["paused"]
    duration=len(targets)/kit["control_hz"] if targets is not None else 3600
    with DrumSim(kit,duration_s=duration) as sim:
        with mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key) as viewer:
            viewer.cam.lookat[:]=[.12,-.03,.30]
            viewer.cam.distance=1.65
            viewer.cam.azimuth=90
            viewer.cam.elevation=-30
            index=1
            print("Drag to orbit; scroll to zoom; Space pauses; close the window to exit.",flush=True)
            while viewer.is_running():
                start=time.monotonic()
                if not state["paused"]:
                    with viewer.lock():
                        if targets is not None:
                            if index>=len(targets): sim.reset(); index=1
                            sim.step(targets[index]); index+=1
                        else:
                            sim.step(sim.last_action)
                viewer.sync()
                time.sleep(max(0,1/kit["control_hz"]/speed-(time.monotonic()-start)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command",choices=["view"])
    parser.add_argument("--demo",help="a robot demonstration folder (data/robot/<take> or data/jaco_v1/<take>)")
    parser.add_argument("--speed",type=float,default=1.)
    args=parser.parse_args()
    view(args.demo,args.speed)
