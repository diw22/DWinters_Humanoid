"""The Unitree G1 with Dex3 hands (third_party/unitree_g1) at the LPD8: model, kinematics and limits.

Fixed pelvis; both 7-joint arms and hands are position-controlled (28 actions). The policy acts in 15 numbers, the 14
arm joints + the right hand's grip (to_policy / from_policy), and sees through a camera in the head.
"""
import copy
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

MODEL = Path(__file__).resolve().parents[1]/"third_party/unitree_g1/g1_with_hands.xml"
SIDES = ("left", "right")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
JOINTS = [f"{s}_{j}_joint" for s in SIDES for j in [*ARM, *(f"hand_{f}" for f in HAND)]]
ACTION_DIM, STATE_DIM = len(JOINTS), 2*len(JOINTS)
ARM_INDICES = np.array([14*s+i for s in range(2) for i in range(7)])
FINGER_INDICES = np.array([14*s+7+i for s in range(2) for i in range(7)])
SPEED = np.tile(np.r_[[37.]*5, 22., 22., 3.14, [12.]*6], 2)  # rad/s, g1_29dof_with_hand URDF
FIXED = ("hip", "knee", "ankle", "waist")
TIPS = {"thumb_tip": "thumb_2", "index_tip": "index_1", "middle_tip": "middle_1"}  # fingertip -> distal link
TIP_GEOMS = {"left_index_tip": "left"}                                             # pad presses
KNOB_FINGER_GEOMS = {"right_thumb_tip": 1, "right_index_tip": 2, "right_middle_tip": 3}
# Hand frame in the wrist_yaw_link: fingers along +x, the index finger on the +z side of the middle one
HAND_AXES = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]]).T   # columns: along the hand, across to the index side, normal


@lru_cache(None)
def _model_info():
    """Joint ranges, and each fingertip (body frame): the distal mesh vertex furthest from its joint."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    ranges = np.array([m.jnt_range[m.joint(j).id] for j in JOINTS])
    tips = {}
    for side in SIDES:
        for tip, link in TIPS.items():
            body = m.body(f"{side}_hand_{link}_link").id
            g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g])
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            v = v@R.reshape(3, 3).T+m.geom_pos[g]
            tips[f"{side}_{tip}"] = v[np.argmax(np.linalg.norm(v, axis=1))]
    return ranges, tips


def action_bounds():
    ranges, _ = _model_info()
    return ranges[:, 0], ranges[:, 1]


def velocity_limits(kit):
    return SPEED


def add_robots(root, world, actuators, kit):
    g1 = ET.parse(MODEL).getroot()
    for tag in ("default", "asset"):
        root.append(copy.deepcopy(g1.find(tag)))
    asset = root.find("asset")
    for node in list(asset):
        if node.get("file"):
            if node.tag == "mesh":  # a 5%-face copy for rendering (assets_decimated/); collisions keep the original
                ET.SubElement(asset, "mesh", name=(node.get("name") or Path(node.get("file")).stem)+"_visual",
                              file=str(MODEL.parent/"assets_decimated"/node.get("file")))
            node.set("file", str(MODEL.parent/"assets"/node.get("file")))
    _, tips = _model_info()
    pelvis = copy.deepcopy(g1.find("worldbody/body"))
    for geom in pelvis.iter("geom"):
        if geom.get("class") == "collision":  # touches the scene (contype 1) but not itself: the simplified meshes overlap
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
        elif geom.get("class") == "visual":
            geom.set("mesh", geom.get("mesh")+"_visual")
    for body in pelvis.iter("body"):
        body.set("gravcomp", "1")
        for child in list(body):
            if child.tag == "freejoint" or (child.tag == "joint" and any(k in child.get("name") for k in FIXED)):
                body.remove(child)
        for side in SIDES:
            for tip, link in TIPS.items():
                if body.get("name") == f"{side}_hand_{link}_link":
                    next(g for g in body.iter("geom") if g.get("class") == "collision").set("name", f"{side}_{tip}")
                    ET.SubElement(body, "site", name=f"{side}_{tip}", pos=" ".join(map(str, tips[f"{side}_{tip}"])), size=".003")
        if body.get("name") == "torso_link":  # head camera, looking down at the hands and the controller
            ET.SubElement(body, "camera", name="policy", pos=" ".join(map(str, kit["head_camera"]["pos_m"])),
                          xyaxes=" ".join(map(str, kit["head_camera"]["xyaxes"])), fovy=str(kit["head_camera"]["fovy"]))
    base = kit["robot_base"]
    pelvis.set("pos", " ".join(map(str, base["position_m"])))
    pelvis.set("quat", f"{np.cos(base['yaw_rad']/2)} 0 0 {np.sin(base['yaw_rad']/2)}")
    world.append(pelvis)
    for joint in JOINTS:
        ET.SubElement(actuators, "position", name=joint[:-6], joint=joint, **{"class": "g1"})


class RobotKinematics:
    def __init__(self, model):
        import mujoco
        self.mj, self.model, self.data = mujoco, model, mujoco.MjData(model)
        ids = [model.joint(j).id for j in JOINTS]
        self.qadr, self.dadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        self.sites = {f"{s}_{t}": model.site(f"{s}_{t}").id for s in SIDES for t in TIPS}
        self.wrists = [model.body(f"{s}_wrist_yaw_link").id for s in SIDES]

    def positions(self, data):
        return data.qpos[self.qadr].copy()

    def velocities(self, data):
        return data.qvel[self.dadr].copy()

    def set_positions(self, data, action):
        data.qpos[self.qadr] = action

    def solve(self, side, q, points, rotation=None, rotation_weight=.05, joint_weights=None, iterations=30, anchor=None):
        """Weighted IK for one side's 14 joints, starting from `q` (28): points = [(site, target xyz, weight)],
        rotation = target hand frame (3x3, HAND_AXES columns in the world) or None; joint_weights (14) penalize moving
        each joint away from `anchor` (28, default `q`). A fixed anchor makes the result a function of the targets
        alone, not of the previous frame."""
        mj, m, d = self.mj, self.model, self.data
        cols = slice(14*SIDES.index(side), 14*SIDES.index(side)+14)
        low, high = (b[cols] for b in action_bounds())
        start = np.asarray(q if anchor is None else anchor, float)[cols]
        weights = np.ones(14) if joint_weights is None else np.asarray(joint_weights)
        d.qpos[self.qadr] = q
        x = np.asarray(q, float)[cols].copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iterations):
            d.qpos[self.qadr[cols]] = x
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            J, e = [], []
            for site, target, w in points:
                mj.mj_jacSite(m, d, jp, None, self.sites[site])
                J.append(w*jp[:, self.dadr[cols]])
                e.append(w*(target-d.site_xpos[self.sites[site]]))
            if rotation is not None:
                wrist = self.wrists[SIDES.index(side)]
                current = d.xmat[wrist].reshape(3, 3)@HAND_AXES
                mj.mj_jacBody(m, d, jp, jr, wrist)
                J.append(rotation_weight*jr[:, self.dadr[cols]])
                e.append(rotation_weight*Rotation.from_matrix(rotation@current.T).as_rotvec())
            J.append(np.diag(weights))
            e.append(weights*(start-x))
            J, e = np.vstack(J), np.concatenate(e)
            step = np.linalg.lstsq(J, e, rcond=None)[0]
            pushed = ((x <= low+1e-9) & (step < 0)) | ((x >= high-1e-9) & (step > 0))
            if pushed.any():  # joints at a limit being pushed past it are frozen, so the others take over this step
                J[:, pushed] = 0
                step = np.linalg.lstsq(J, e, rcond=None)[0]
                step[pushed] = 0
            x = np.clip(x+np.clip(step, -.1, .1), low, high)  # capped steps: conflicting rows can't throw the arm around
        out = np.asarray(q, float).copy()
        out[cols] = x
        return out

    def site(self, data, name):
        return data.site_xpos[self.sites[name]].copy()


def acceleration_limits(robot, kit):
    """Each joint's torque-limited acceleration (rad/s^2) at the rest pose: torque limit / joint-space inertia
    (link inertia + motor armature), times kit["torque_margin"] (torque is also needed for damping and coupling)."""
    m, d = robot.model, robot.data
    robot.set_positions(d, rest_action(robot, kit))
    robot.mj.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    robot.mj.mj_fullM(m, M, d.qM)
    ids = [m.joint(j).id for j in JOINTS]
    return np.array([m.jnt_actfrcrange[i][1]/M[m.jnt_dofadr[i], m.jnt_dofadr[i]] for i in ids])*kit["torque_margin"]


def servo_lag(model):
    """Each joint's position-servo lag (s): following a moving command, a PD servo trails it by kv/kp."""
    return np.array([-model.actuator(j[:-6]).biasprm[2]/model.actuator(j[:-6]).gainprm[0] for j in JOINTS])


def rest_action(robot, kit):
    """Both index fingertips at the park spots beside the controller (retarget.tool_park), hands in the press shape and
    the press orientation for that height."""
    from .retarget import tool_park
    q = np.zeros(ACTION_DIM)
    q[FINGER_INDICES] = finger_shape(kit["press_fingers_rad"])
    park = tool_park(kit)
    for side, xyz in zip(SIDES, (park[:3], park[4:7])):
        for _ in range(10):
            q = robot.solve(side, q, [(f"{side}_index_tip", xyz, 1.)], rotation=hand_rotation(kit, side), rotation_weight=.3,
                            joint_weights=np.r_[[.01]*7, [10.]*7], iterations=40)
    return q


def finger_shape(left):
    """Finger angles for both hands (14) from the left hand's 7 (the right hand's joints are mirrored)."""
    return np.clip(np.r_[left, -np.asarray(left)], *(b[FINGER_INDICES] for b in action_bounds()))


# The policy's 15 actions: the 14 arm joints and the right hand's grip. The fingers only ever take fixed shapes: the
# left hand the press shape, the right hand moves along press shape (grip 0) -> claw pre-shape (0.5) -> claw closed (1)
# (retarget.knob_claw), so one number gives all seven.
POLICY_DIM = 15


def _grip_path(kit):
    return np.array([finger_shape(kit["press_fingers_rad"])[7:], kit["knob_claw"]["preshape_fingers_rad"], kit["knob_claw"]["closed_fingers_rad"]])


def to_policy(actions, kit):
    """Robot actions (frames, 28) -> policy actions (frames, 15): the right fingers projected onto the grip path."""
    actions, path = np.atleast_2d(actions), _grip_path(kit)
    fingers, best = actions[:, 21:28], np.full((len(actions), 2), np.inf)  # (distance, grip)
    for k in range(2):
        d = path[k+1]-path[k]
        u = np.clip((fingers-path[k])@d/(d@d), 0, 1)
        dist = np.linalg.norm(fingers-(path[k]+u[:, None]*d), axis=1)
        best = np.where((dist < best[:, 0])[:, None], np.c_[dist, (k+u)/2], best)
    return np.c_[actions[:, ARM_INDICES], best[:, 1]].astype(np.float32)


def from_policy(policy, kit):
    """Policy actions (..., 15) -> robot actions (..., 28)."""
    policy, path = np.asarray(policy, float), _grip_path(kit)
    g = np.clip(policy[..., 14:15]*2, 0, 2)
    k = np.minimum(np.floor(g), 1)
    right = path[k.astype(int)[..., 0]]+(g-k)*(path[k.astype(int)[..., 0]+1]-path[k.astype(int)[..., 0]])
    out = np.zeros(policy.shape[:-1]+(ACTION_DIM,))
    out[..., ARM_INDICES] = policy[..., :14]
    out[..., 7:14] = finger_shape(kit["press_fingers_rad"])[:7]
    out[..., 21:28] = right
    return out


def hand_rotation(kit, side):
    """The one hand orientation (HAND_AXES columns in the world): the human's median pressing pose, kit["press_hand_deg"]
    = pitch (fingers down; negative = knuckles above the wrist), yaw (turned toward the controller's middle) and roll
    (index side up), mirrored for the right hand. The human keeps it while the finger lifts (palm pitch changes ~6
    degrees per 10 cm), so it is fixed: the same fingertip position always gives the same arm pose."""
    pitch, yaw, roll = np.radians([kit["press_hand_deg"][k] for k in ("pitch", "yaw", "roll")])
    along = np.array([np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), -np.sin(pitch)])
    level = np.array([np.cos(yaw), -np.sin(yaw), 0.])   # horizontal, across the hand toward the index side
    across = np.cos(roll)*level+np.sin(roll)*np.cross(level, along)
    if side == "right":
        along, across = along*[-1, 1, 1], across*[-1, 1, 1]
    return np.c_[along, across, np.cross(along, across)]
