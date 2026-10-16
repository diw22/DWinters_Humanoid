"""The v1 robot, two Kinova Jaco arms (third_party/dm_control/kinova): model, IK and tool-path planner.
Used by the v1 demonstrations in data/jaco_v1.
"""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ACTION_DIM=18
STATE_DIM=36
ARM_INDICES=np.array([0,1,2,3,4,5,9,10,11,12,13,14])
FINGER_INDICES=np.array([6,7,8,15,16,17])
MODEL_DIR=Path(__file__).resolve().parents[1]/"third_party/dm_control/kinova"
# Finite software envelope for continuous joints; bounded joints use upstream limits.
LOWER=np.array([-2*np.pi,.820305,.331613,-2*np.pi,-2*np.pi,-2*np.pi])
UPPER=np.array([2*np.pi,5.462881,5.951573,2*np.pi,2*np.pi,2*np.pi])
REST=np.array([0.,4.,4.,0.,1.5,0.])
OPEN=np.array([.15,.15,.15])
PRESS_FINGERS=np.array([1.,1.35,1.35])
PRESS_TCP=np.array([.00462288,.03233752,-.20274023])
TIP_GEOMS={"left_finger_tip_1":"left"}                                   # pad presses
KNOB_FINGER_GEOMS={f"right_finger_tip_{i}":i for i in (1,2,3)}


def action_bounds():
    return np.tile(np.r_[LOWER,[.15]*3],2),np.tile(np.r_[UPPER,[1.35]*3],2)


def velocity_limits(kit):
    # dm_control JacoArm firmware velocity limits, with an additional project scale.
    # Finger speed is a conservative project limit; no real hardware controller is provided.
    return np.tile(np.r_[np.deg2rad([36.]*3+[48.]*3),[2.]*3],2)*kit["joint_velocity_scale"]


def initial_fingers(kit):
    return np.array([PRESS_FINGERS,kit["knob_grasp"]["pregrasp_fingers_rad"]])


def rest_action(robot,kit):
    from .retarget import tool_park
    return robot.solve_pair(tool_park(kit),initial_fingers(kit))


def add_robots(root,world,actuators,kit):
    arm_xml=ET.parse(MODEL_DIR/"jaco_arm.xml").getroot()
    hand_xml=ET.parse(MODEL_DIR/"jaco_hand.xml").getroot()
    common=ET.parse(MODEL_DIR/"common.xml").getroot()
    defaults=copy.deepcopy(common.find("default"));root.append(defaults)
    assets=copy.deepcopy(common.find("asset"));root.append(assets)
    for source,name in ((arm_xml,"jaco_arm"),(hand_xml,"jaco_hand")):
        default=copy.deepcopy(source.find("default"));default.set("class",name)
        default.find("geom").set("contype","1");default.find("geom").set("conaffinity","1")
        defaults.append(default)
        for mesh in source.find("asset"):
            mesh=copy.deepcopy(mesh)
            mesh.set("file",str((MODEL_DIR/"meshes_decimated"/mesh.get("file")).resolve()))
            assets.append(mesh)
    bases=kit["robot_bases"]
    contacts=ET.SubElement(root,"contact")
    for index,arm in enumerate(("left","right")):
        def prefixed(element):
            item=copy.deepcopy(element)
            for node in item.iter():
                for attribute in ("name","joint","body1","body2"):
                    if attribute in node.attrib: node.set(attribute,arm+"_"+node.get(attribute))
            return item
        body=prefixed(arm_xml.find("worldbody/body"));body.set("childclass","jaco_arm")
        body.set("pos"," ".join(map(str,bases[index]["position_m"])))
        yaw=bases[index]["yaw_rad"]
        body.set("quat",f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}")
        hand=prefixed(hand_xml.find("worldbody/body"));hand.set("childclass","jaco_hand")
        wrist=ET.SubElement(body.find(f".//body[@name='{arm}_b_6']"),"body",
                            name=f"{arm}_wrist_attachment",quat="0 .70710678118 .70710678118 0")
        wrist.append(hand)
        for b in body.iter("body"): b.set("gravcomp","1")
        if index==0:
            # Lowest point of the unscaled upstream finger-1 mesh at PRESS_FINGERS.
            # The actual mesh makes contact; no artificial striker geometry is added.
            tcp=PRESS_TCP
        else:
            tcp=[.015,.0024,-.193]
        ET.SubElement(hand,"site",name=f"{arm}_tcp",pos=" ".join(map(str,tcp)),
                      quat="0 .70710678118 .70710678118 0",size=".002",rgba="1 .5 0 1")
        world.append(body)
        for source in (arm_xml,hand_xml):
            for entry in source.find("contact"): contacts.append(prefixed(entry))
        speedup=kit["joint_velocity_scale"]/.8  # Jaco torques, scaled with the sim speed-up (kit limits_note)
        for j in range(1,7):
            torque=(30.5 if j<=3 else 6.8)*speedup
            ET.SubElement(actuators,"position",name=f"{arm}_motor_{j}",joint=f"{arm}_joint_{j}",
                          kp="6000",kv="180",ctrlrange=f"{LOWER[j-1]} {UPPER[j-1]}",forcerange=f"{-torque} {torque}")
        stiff=speedup if arm=="left" else 1  # the pressing finger must not fold when it rests on a pad
        for j in range(1,4):
            ET.SubElement(actuators,"position",name=f"{arm}_finger_motor_{j}",joint=f"{arm}_finger_{j}",
                          kp=f"{12*stiff}",kv=f"{1.5*stiff}",ctrlrange=".15 1.35",forcerange=f"{-stiff} {stiff}")


class RobotKinematics:
    def __init__(self,model):
        import mujoco
        self.mj=mujoco; self.model=model; self.data=mujoco.MjData(model)
        self.joints=[]; self.qindices=[]; self.dindices=[]; self.fingers=[]; self.sites=[]
        for arm in ("left","right"):
            joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"{arm}_joint_{i}") for i in range(1,7)]
            fingers=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"{arm}_finger_{i}") for i in (1,2,3)]
            self.joints.append(joints)
            self.qindices.append(model.jnt_qposadr[joints])
            self.dindices.append(model.jnt_dofadr[joints])
            self.fingers.append(model.jnt_qposadr[fingers])
            self.sites.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f"{arm}_tcp"))

    def positions(self,data):
        return np.concatenate([np.r_[data.qpos[q],data.qpos[f]] for q,f in zip(self.qindices,self.fingers)])

    def velocities(self,data):
        # All joints are hinge/slide, so nq == nv and finger qpos/dof indices coincide.
        return np.concatenate([np.r_[data.qvel[q],data.qvel[f]] for q,f in zip(self.dindices,self.fingers)])

    def set_positions(self,data,action):
        for arm in range(2):
            data.qpos[self.qindices[arm]]=action[9*arm:9*arm+6]
            data.qpos[self.fingers[arm]]=action[9*arm+6:9*arm+9]

    def pose(self,data,arm):
        site=self.sites[arm]
        return data.site_xpos[site].copy(),data.site_xmat[site].reshape(3,3).copy()

    def solve(self,arm,xyz,yaw,seed):
        q=np.clip(np.asarray(seed,dtype=float).copy(),LOWER+1e-5,UPPER-1e-5)
        target_rotation=Rotation.from_euler("z",yaw+(np.pi if arm else 0.)).as_matrix() @ np.diag([1.,-1.,-1.])
        jp=np.zeros((3,self.model.nv)); jr=np.zeros_like(jp)
        for _ in range(220):
            self.data.qpos[self.qindices[arm]]=q
            self.mj.mj_kinematics(self.model,self.data)
            self.mj.mj_comPos(self.model,self.data)
            position,rotation=self.pose(self.data,arm)
            ep=np.asarray(xyz)-position
            er=Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
            if np.linalg.norm(ep)<.000002 and np.linalg.norm(er)<.00002: return q
            self.mj.mj_jacSite(self.model,self.data,jp,jr,self.sites[arm])
            jac=np.vstack([jp[:,self.dindices[arm]],.25*jr[:,self.dindices[arm]]])
            error=np.r_[ep,.25*er]
            delta=jac.T @ np.linalg.solve(jac@jac.T+np.eye(6)*1e-5,error)
            delta*=min(1.,.12/max(np.max(abs(delta)),1e-9))
            q=np.clip(q+delta,LOWER+1e-5,UPPER-1e-5)
        raise ValueError(f"Jaco {'left' if arm==0 else 'right'} IK failed: position {np.linalg.norm(ep):.4f}m, rotation {np.linalg.norm(er):.3f}rad")

    def solve_pair(self,tool_targets,fingers,seed=None):
        initial = seed is None
        if seed is None: seed=np.tile(np.r_[REST,OPEN],2)
        result=[]
        for arm in range(2):
            target=tool_targets[4*arm:4*arm+4]
            seeds=[seed[9*arm:9*arm+6]]
            if initial:
                rng=np.random.default_rng(7)
                seeds += [rng.uniform(LOWER+.1,UPPER-.1) for _ in range(100)]
            for attempt in seeds:
                try:
                    q=self.solve(arm,target[:3],target[3],attempt)
                    if initial:
                        self.data.qpos[self.qindices[arm]]=q
                        self.data.qpos[self.fingers[arm]]=fingers[arm]
                        self.mj.mj_forward(self.model,self.data)
                        prefix=("left_","right_")[arm]
                        colliding=False
                        for c in self.data.contact:
                            names=[self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_GEOM,g) or "" for g in (c.geom1,c.geom2)]
                            # Left is solved first; right must also clear the solved left arm.
                            ignore_unsolved=arm==0 and any(n.startswith("right_") for n in names)
                            if any(n.startswith(prefix) for n in names) and not ignore_unsolved and c.dist<-.0005:
                                colliding=True;break
                        if colliding: continue
                    break
                except ValueError:
                    if not initial: raise
            else: raise ValueError(f"Jaco {arm} cannot reach initial pose")
            result.extend(np.r_[q,fingers[arm]])
        return np.asarray(result)


# ---------------------------------------------------------------- planning (tool paths -> joints)

def tool_plan(events, kit, duration, path):
    """Left tool = `path` (fingertip xyz per control step, robot frame); right tool = a scripted pinch-turn per
    "turn" event: approach, close, turn, release, lift, reset the wrist above the controls."""
    from .retarget import control_times,controls,smoothstep5,target_xyz,tool_park
    times=control_times(kit,duration)
    fps=kit["control_hz"]
    lookup=controls(kit)
    grasp=kit["knob_grasp"]
    pre,closed=np.array(grasp["pregrasp_fingers_rad"]),np.array(grasp["closed_fingers_rad"])
    angles={p["id"]:p["initial_angle_rad"] for p in kit["controls"] if p["kind"]=="knob"}
    knots=[(0.,np.r_[tool_park(kit)[4:],pre])]
    for e in sorted((e for e in events if e["kind"]=="turn"),key=lambda e:e["time_s"]):
        t=e["time_s"]
        delta=e["angle_rad"]-angles[e["pad"]]
        angles[e["pad"]]=e["angle_rad"]
        if abs(delta)>grasp["max_turn_rad"]+1e-8:
            raise ValueError("Knob turn exceeds the validated pinch arc (0.4 rad). Record smaller turns with a pause for release/regrasp.")
        touch=np.r_[target_xyz(lookup[e["pad"]],kit),0.,pre]
        hover=touch.copy();hover[2]+=grasp["clearance_m"]
        grip=touch.copy();grip[4:]=closed
        turned=grip.copy();turned[3]=delta
        released=turned.copy();released[4:]=grasp["release_fingers_rad"]
        raised=released.copy();raised[2]+=grasp["clearance_m"]
        # Release before lifting; reset wrist yaw only above all controls.
        reset=raised.copy();reset[3]=0.;reset[4:]=pre
        approach,finish=t-grasp["close_s"]-grasp["approach_s"],t+e["duration_s"]
        if approach<=knots[-1][0]+1/fps:
            raise ValueError(f"Too little travel time before the turn of {e['pad']} at {t:.3f}s; allow time for approach/grasp/release")
        up=finish+grasp["release_s"]+grasp["retract_s"]
        knots+=[(approach,hover),(t-grasp["close_s"],touch),(t,grip),(finish,turned),
                (finish+grasp["release_s"],released),(up,raised),(up+grasp["reset_s"],reset)]
    if knots[-1][0]>=duration:
        raise ValueError("Leave enough time after the final turn for release and retract (at least 4 seconds)")
    knots.append((duration,knots[-1][1].copy()))
    right=np.empty((len(times),7))
    for (t0,q0),(t1,q1) in zip(knots[:-1],knots[1:]):
        mask=(times>=t0)&(times<=t1)
        right[mask]=q0+(q1-q0)*smoothstep5(np.clip((times[mask]-t0)/(t1-t0),0,1))[:,None]
    right[times>duration]=knots[-1][1]
    tools=np.c_[path,np.zeros(len(times)),right[:,:4]]
    fingers=np.stack([np.tile(PRESS_FINGERS,(len(times),1)),right[:,4:]],axis=1)
    velocity=abs(np.diff(tools,axis=0))*fps
    limits=np.tile([kit["max_linear_speed_m_s"]]*3+[kit["max_angular_speed_rad_s"]],2)
    if np.any(velocity>limits+1e-6):
        raise ValueError("Tool trajectory too fast; slow the original performance")
    return times,tools,fingers


def smooth_joints(reference,kit,pins):
    """Jaco: the arm joints only (finger commands are kept); the knob arm gets a much tighter corridor and no pins."""
    from .retarget import smooth
    settings=kit["motion_planner"]
    amax=np.tile(np.r_[[settings["max_arm_acceleration_rad_s2"]]*6,[8.]*3],2)
    tube={j:settings["max_joint_deviation_rad"] if j<9 else .0002 for j in ARM_INDICES}
    return smooth(reference,kit,ARM_INDICES,action_bounds(),velocity_limits(kit),amax,tube,{j:pins if j<9 else [] for j in ARM_INDICES})


def plan(events,kit,duration,path):
    import mujoco
    from .sim import scene_xml
    times,tools,fingers=tool_plan(events,kit,duration,path)
    model=mujoco.MjModel.from_xml_string(scene_xml(kit))
    robot=RobotKinematics(model)
    joints=[]; seed=None
    for target,grasp in zip(tools,fingers):
        seed=robot.solve_pair(target,grasp,seed)
        joints.append(seed.copy())
    pins=[int(round(e["time_s"]*kit["control_hz"])) for e in events if e["kind"]!="turn"]
    joints,adjustment=smooth_joints(np.array(joints),kit,pins)
    deviations=np.zeros((len(joints),2))
    for i,action in enumerate(joints):
        robot.set_positions(robot.data,action)
        mujoco.mj_kinematics(model,robot.data)
        for arm in range(2):
            deviations[i,arm]=np.linalg.norm(robot.pose(robot.data,arm)[0]-tools[i,4*arm:4*arm+3])
    # 3 mm: human fingertip paths are fast; presses themselves are pinned exactly.
    if np.any(deviations>np.array([.003,.0003])):
        raise ValueError(f"QP smoothing deviated too far from the contact path ({deviations.max(axis=0).round(4).tolist()} m); allow more travel time")
    return times,joints,{"max_joint_adjustment_rad":adjustment,"max_tool_deviation_m":deviations.max(axis=0).tolist()}
