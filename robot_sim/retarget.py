"""Fingertip path -> G1 joint targets: per-frame IK, then joint smoothing (OSQP) with every hit pinned (plan_g1);
knob takes use a claw grip (knob_claw). Also the scene helpers shared by both robots.
"""
import numpy as np
import osqp
from scipy import sparse

from .common import board_to_robot


def controls(kit):
    return {p["id"]:p for p in kit["pads"] + kit["controls"]}


def surface_offset(control,kit):
    """Height of a control's top above pad contact height (the board origin), shared by planning and physics."""
    geometry=kit["enclosure"]
    field={"pad":"pad_surface_height_m","button":"button_surface_height_m","knob":"knob_top_height_m"}[control.get("kind","pad")]
    return geometry[field]-geometry["pad_surface_height_m"]


def target_xyz(control, kit):
    xy = control.get("center_m")
    if xy is None:
        x0,y0,x1,y1 = control["bounds_m"]
        xy = [(x0+x1)/2,(y0+y1)/2]
    p = board_to_robot(xy,kit)
    p[2] += surface_offset(control,kit)
    p[2] -= kit["knob_grasp"]["tcp_below_top_m"] if control.get("kind")=="knob" else kit["press_depth_m"]
    return p


def tool_park(kit):
    return np.array([np.r_[board_to_robot(xy,kit)+[0,0,
                         kit["hover_height_m"] if i==0 else .06],0.]
                     for i,xy in enumerate(kit["park_xy_m"])]).reshape(8)


def tool_bounds(kit):
    x0,y0,x1,y1 = kit["workspace_xy_m"]
    base = np.array(kit["board_origin_robot_m"])
    low = np.r_[base+[x0,y0,-.004],-np.pi]
    high = np.r_[base+[x1,y1,.20],np.pi]
    return np.tile(low,2),np.tile(high,2)


def control_times(kit,duration):
    return np.arange(int(np.ceil(duration*kit["control_hz"]))+1)/kit["control_hz"]


def smoothstep5(u):
    """Zero velocity and acceleration at both ends of a move."""
    return u*u*u*(10+u*(-15+6*u))


def smooth(reference,kit,joints,bounds,vmax,amax,tube,pins):
    """Minimize tracking error plus squared acceleration/jerk (a sparse QP per joint in `joints`). Hard bounds: joint
    range, velocity, acceleration and a corridor of tube[j] around the IK path. Both ends and the frames in pins[j]
    are held exactly; other columns are kept."""
    n,hz,settings=len(reference),kit["control_hz"],kit["motion_planner"]
    eye=sparse.eye(n,format="csc")
    d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format="csc")
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format="csc")
    d3=sparse.diags([-np.ones(n-3),3*np.ones(n-3),-3*np.ones(n-3),np.ones(n-3)],[0,1,2,3],shape=(n-3,n),format="csc")
    # Differences are per frame in the objective; constraint bounds use seconds.
    cost=2*(eye+settings["acceleration_weight"]*(d2.T@d2)+settings["jerk_weight"]*(d3.T@d3))
    constraints=sparse.vstack([eye,d1,d2],format="csc")
    low,high=bounds
    result=reference.copy()
    for j in joints:
        lower=np.maximum(reference[:,j]-tube[j],low[j]);upper=np.minimum(reference[:,j]+tube[j],high[j])
        fixed=np.unique(np.r_[0,n-1,pins[j]]).astype(int)
        lower[fixed]=reference[fixed,j];upper[fixed]=reference[fixed,j]
        vlow=np.full(n-1,-vmax[j]/hz);vhigh=-vlow
        # Start and finish at rest. Keeps acceleration finite at reset/hold.
        vlow[[0,-1]]=0.;vhigh[[0,-1]]=0.
        solver=osqp.OSQP()
        solver.setup(P=sparse.triu(cost,format="csc"),q=-2*reference[:,j],A=constraints,
                     l=np.r_[lower,vlow,np.full(n-2,-amax[j]/hz**2)],
                     u=np.r_[upper,vhigh,np.full(n-2,amax[j]/hz**2)],
                     verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=15000,polishing=True)
        solver.warm_start(x=reference[:,j])
        solution=solver.solve(raise_error=False)
        if solution.info.status_val!=1:
            raise ValueError(f"Motion QP cannot satisfy joint {j+1} limits ({solution.info.status}); allow more travel time")
        result[:,j]=solution.x
    return result,float(np.max(abs(result-reference)))


def g1_reference(events,kit,duration,path,side="left"):
    """The per-frame IK part of plan_g1: control times, joints (frames, 28), the kinematics and the hit frames."""
    import mujoco
    from . import g1
    from .sim import scene_xml
    robot=g1.RobotKinematics(mujoco.MjModel.from_xml_string(scene_xml(kit)))
    times=control_times(kit,duration)
    q=g1.rest_action(robot,kit)
    anchor=q.copy()
    weights=np.r_[[1e-3]*7,[10.]*7]  # arm: pulled lightly toward the rest posture; fingers: held in the press shape
    pins=[int(round(e["time_s"]*kit["control_hz"])) for e in events]
    rotation=g1.hand_rotation(kit,side)
    joints=[]
    for target in path:
        q=robot.solve(side,q,[(f"{side}_index_tip",target,1.)],rotation,.3,weights,iterations=30,anchor=anchor)
        joints.append(q.copy())
    return times,np.array(joints),robot,pins


def knob_claw(kit,times,angle):
    """Right-hand joints (frames, 28) and the right index tip path that turn kit["knob_claw"]["knob"] through `angle`
    (rad per control step, the sim's convention: 0 = rest, negative = clockwise) with the claw grip of kit["knob_claw"]:
    above the knob in the pre-shape 1.2-2.6 s before the first turn (a joint-space move from rest), straight down,
    squeeze, the hand turning about the knob axis with the angle (clipped to 0..-54 deg, the range checked for reach and
    clearance), then open, up and back to rest 0.2-2.6 s after the last turn. Rest outside that."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from . import g1
    from .sim import scene_xml
    claw=kit["knob_claw"];knob=controls(kit)[claw["knob"]]
    robot=g1.RobotKinematics(mujoco.MjModel.from_xml_string(scene_xml(kit)));data=robot.data
    rest=g1.rest_action(robot,kit);joints=np.tile(rest,(len(times),1))
    robot.set_positions(data,rest);mujoco.mj_kinematics(robot.model,data);path=np.tile(robot.site(data,"right_index_tip"),(len(times),1))
    turning=np.flatnonzero(angle<-np.radians(3))  # clockwise (the only way that changes the sound), beyond the camera's noise at rest
    if not len(turning):
        return joints,path
    first,last=times[turning[0]],times[turning[-1]]
    if first<2.6 or last>times[-1]-2.6:
        raise ValueError(f"Knob turned at {first:.1f}-{last:.1f} s: no time to reach it from rest and back")
    top=board_to_robot(knob["center_m"],kit)+[0,0,surface_offset(knob,kit)]
    Rk,pk=Rotation.from_rotvec(claw["wrist_rotvec"]).as_matrix(),np.array(claw["wrist_offset_m"])
    closed,open_=np.array(claw["closed_fingers_rad"]),np.array(claw["preshape_fingers_rad"])
    lift=claw["lift_m"]*(1-smoothstep5(np.interp(times,[first-1.2,first-.6,last+.6,last+1.2],[0,1,1,0])))
    squeeze=smoothstep5(np.interp(times,[first-.6,first-.2,last+.2,last+.6],[0,1,1,0]))
    theta=np.clip(angle,np.radians(-54),0)
    wrist=robot.wrists[1];q=rest.copy()
    near=np.flatnonzero((times>=first-1.2)&(times<=last+1.2))
    for n,i in enumerate(near):
        Rz=Rotation.from_euler("z",theta[i]).as_matrix();Rw,pw=Rz@Rk,top+Rz@pk+[0,0,lift[i]]
        fingers=open_+(closed-open_)*squeeze[i]
        q=np.r_[q[:21],fingers];robot.set_positions(data,q);mujoco.mj_kinematics(robot.model,data)
        R,p=data.xmat[wrist].reshape(3,3),data.xpos[wrist]
        tips=[(s,pw+Rw@(R.T@(robot.site(data,s)-p)),1.) for s in ("right_index_tip","right_thumb_tip")]
        for _ in range(10 if n==0 else 1):
            q=robot.solve("right",q,tips,Rw@g1.HAND_AXES,.3,np.r_[[1e-3]*7,[10.]*7],iterations=40,anchor=np.r_[rest[:21],fingers])
        q[21:28]=fingers;joints[i]=q;path[i]=tips[0][1]
    for edge,(t0,t1) in ((near[0],(first-2.6,first-1.2)),(near[-1],(last+2.6,last+1.2))):  # joint-space moves from/to rest
        mask=(times>t0)&(times<t1) if t0<t1 else (times>t1)&(times<t0)
        w=smoothstep5((times[mask]-t0)/(t1-t0))[:,None]
        joints[mask]=rest+(joints[edge]-rest)*w
        for i in np.flatnonzero(mask):
            robot.set_positions(data,joints[i]);mujoco.mj_kinematics(robot.model,data);path[i]=robot.site(data,"right_index_tip")
    return joints,path


def plan_g1(events,kit,duration,path,side="left",reference=None):
    """G1: the `side` index fingertip follows `path` (robot frame, per control step) with one fixed press hand: the
    fingers hold kit["press_fingers_rad"] (index bent to point ~48 degrees down like the human's, middle and thumb
    tucked) and the hand keeps the human's median pressing orientation (g1.hand_rotation), so only the 7 arm joints
    move and their one spare degree of freedom is pulled toward the rest posture. The same fingertip position always gives the same joints:
    consistent demonstrations, whatever the human hand did. The other hand stays at rest. Then the joint QP with the
    G1's real speed limits and torque-limited accelerations (hits pin the left side). How far the index tip ends up
    from its path is reported, not enforced: the physics replay decides."""
    import mujoco
    from . import g1
    if reference is None:
        times,joints,robot,pins=g1_reference(events,kit,duration,path,side)
    else:  # joints planned elsewhere (knob_claw)
        from .sim import scene_xml
        times,joints,pins=control_times(kit,duration),reference,[]
        robot=g1.RobotKinematics(mujoco.MjModel.from_xml_string(scene_xml(kit)))
    model=robot.model
    settings=kit["motion_planner"]
    amax=g1.acceleration_limits(robot,kit)
    tube=np.where(np.isin(np.arange(g1.ACTION_DIM),g1.ARM_INDICES),settings["max_joint_deviation_rad"],settings["max_finger_deviation_rad"])
    joints,adjustment=smooth(joints,kit,range(g1.ACTION_DIM),g1.action_bounds(),g1.velocity_limits(kit),amax,tube,
                             [pins if j<14 else [] for j in range(g1.ACTION_DIM)])
    deviation=np.zeros(len(times))
    for i,action in enumerate(joints):
        robot.set_positions(robot.data,action)
        mujoco.mj_kinematics(model,robot.data)
        deviation[i]=np.linalg.norm(robot.site(robot.data,f"{side}_index_tip")-path[i])
    near=np.zeros(len(times),bool)
    for p in pins:
        near[max(0,p-int(.25*kit["control_hz"])):p+int(.25*kit["control_hz"])+1]=True
    # Each joint's commands go out early by its servo's lag (kv/kp, 9-41 ms), so the joints are where the plan says
    # when it says: without it hits landed ~25 ms late and fast strokes turned back ~5 mm above the pad. Same
    # trajectory, shifted in time: speeds and accelerations unchanged.
    lag=g1.servo_lag(model)
    joints=np.stack([np.interp(times+lag[j],times,joints[:,j]) for j in range(g1.ACTION_DIM)],1)
    return times,joints,{"max_joint_adjustment_rad":adjustment,"max_tip_deviation_near_hits_m":float(deviation[near].max()) if near.any() else 0.,
                         "max_tip_deviation_m":float(deviation.max())}


def retarget(events, kit, duration, path, side="left", reference=None):
    """Fingertip path (robot frame, per control step; G1: the `side` index tip, `reference`: joints already planned
    for it) -> planned joint targets (time_s, joints) and the planning report, for the kit's robot."""
    if kit.get("robot")=="g1":
        return plan_g1(events["events"],kit,duration,path,side,reference)
    from .jaco import plan  # the two Jaco arms of the v1 demonstrations
    return plan(events["events"],kit,duration,path)
