"""The drum-kit scene with the robot, its joint motors and contact-triggered pad sounds."""
import xml.etree.ElementTree as ET
import numpy as np

from .common import board_to_robot
from .retarget import controls, surface_offset


def robot_module(kit):
    """The robot a config uses: the Unitree G1 ("robot": "g1", robot_sim/g1.py) or the two Jaco arms (robot_sim/jaco.py:
    configs without "robot", like the v1 demonstrations in data/jaco_v1)."""
    from . import g1, jaco
    return g1 if kit.get("robot")=="g1" else jaco


def vector(x):
    return " ".join(str(float(v)) for v in x)


def pad_sample(control,rate):
    """A decaying tone per pad; the LPD8 mode buttons are silent."""
    if control.get("kind")=="button":
        return np.zeros(1,dtype=np.float32)
    t=np.arange(round(rate*.18))/rate
    return (.4*np.sin(2*np.pi*control["frequency_hz"]*t)*np.exp(-25*t)).astype(np.float32)


def add_enclosure(world,kit):
    """The controller body (collides) with a slightly narrower visual case and red end caps."""
    base=np.asarray(kit["board_origin_robot_m"])
    geometry=kit["enclosure"]
    width,depth,_=geometry["size_m"]
    body_height=geometry["body_height_m"]
    bottom=-geometry["pad_surface_height_m"]
    center=base+[width/2,depth/2,bottom+body_height/2]
    ET.SubElement(world,"geom",name="controller_enclosure",type="box",pos=vector(center),
                  size=vector([width/2,depth/2,body_height/2]),rgba=".12 .13 .14 1",group="3",friction=".7 .01 .001")
    ET.SubElement(world,"geom",name="controller_case_visual",type="box",pos=vector(center),
                  size=vector([width/2-.003,depth/2,body_height/2]),rgba=".12 .13 .14 1",contype="0",conaffinity="0")
    for side,x in (("left",.0015),("right",width-.0015)):
        ET.SubElement(world,"geom",name=f"controller_{side}_red_endcap",type="box",
                      pos=vector(base+[x,depth/2,bottom+body_height/2]),size=vector([.0015,depth/2,body_height/2]),
                      rgba=".72 .025 .035 1",contype="0",conaffinity="0")


def scene_xml(kit):
    root=ET.Element("mujoco",model="lpd8")
    ET.SubElement(root,"compiler",angle="radian",autolimits="true")
    ET.SubElement(root,"option",timestep=str(1/(kit["control_hz"]*10)),gravity="0 0 -9.81",integrator="implicitfast",cone="elliptic",iterations="100")
    visual=ET.SubElement(root,"visual"); ET.SubElement(visual,"global",offwidth="1280",offheight="1280")
    world=ET.SubElement(root,"worldbody")
    ET.SubElement(world,"light",pos="0 -1 2.5",diffuse="0.8 0.8 0.8")
    ET.SubElement(world,"light",pos="1 1 2",diffuse="0.5 0.5 0.5")
    base=np.asarray(kit["board_origin_robot_m"])
    if "head_camera" not in kit:  # the G1 carries its policy camera in its head
        ET.SubElement(world,"camera",name="policy",pos=vector(base+[.15,-.20,.36]),xyaxes="1 0 0 0 .8 .6",fovy="48")
    ET.SubElement(world,"camera",name="overview",pos=vector(base+[.12,-1.3,.83]),xyaxes="1 0 0 0 .45 .893",fovy="48")
    ET.SubElement(world,"geom",name="floor",type="plane",size="2 2 .01",rgba=".16 .19 .23 1")
    if "desk_size_m" in kit:  # a desk under the controller, from the floor to the controller's bottom
        top=base[2]-kit["enclosure"]["pad_surface_height_m"]
        width,depth=kit["desk_size_m"]
        ET.SubElement(world,"geom",name="desk",type="box",pos=vector(base+[kit["enclosure"]["size_m"][0]/2,depth/2-.1,-top/2-kit["enclosure"]["pad_surface_height_m"]]),
                      size=vector([width/2,depth/2,top/2]),rgba=".35 .3 .26 1")
    add_enclosure(world,kit)
    actuator=ET.SubElement(root,"actuator")
    robot_module(kit).add_robots(root,world,actuator,kit)
    for p in controls(kit).values():
        if p.get("kind")=="knob":
            top=surface_offset(p,kit)
            bottom=kit["enclosure"]["body_height_m"]-kit["enclosure"]["pad_surface_height_m"]
            half_height=(top-bottom)/2
            pos=board_to_robot(p["center_m"],kit)+[0,0,(top+bottom)/2]
            body=ET.SubElement(world,"body",name=p["id"],pos=vector(pos))
            ET.SubElement(body,"joint",name=f"knob_{p['id']}",type="hinge",axis="0 0 1",range="-2.3562 2.3562",damping=".0005",frictionloss=".0001")
            ET.SubElement(body,"geom",name=f"control_{p['id']}",type="cylinder",size=f"{p['radius_m']} {half_height}",
                          rgba=".09 .1 .12 1",mass=".02",group="3",
                          condim="4",friction="1 .005 .0001",solref=".003 1")
            ET.SubElement(body,"geom",name=f"cap_{p['id']}",type="cylinder",size=f"{p['radius_m']} {half_height-.0005}",
                          rgba=".09 .1 .12 1",mass="0",contype="0",conaffinity="0")
            # Inset pointer remains below the configured maximum overall height.
            ET.SubElement(body,"geom",type="box",pos=vector([p["radius_m"]*.45,0,half_height-.0003]),
                          size=vector([p["radius_m"]*.45,.00065,.0003]),rgba=".85 .9 .95 1",contype="0",conaffinity="0")
        else:
            x0,y0,x1,y1=p["bounds_m"]
            top=surface_offset(p,kit)
            half_height=.001
            pos=board_to_robot([(x0+x1)/2,(y0+y1)/2],kit)+[0,0,top-half_height]
            ET.SubElement(world,"geom",name=f"rim_{p['id']}",type="box",pos=vector(pos+[0,0,half_height-.0006]),
                          size=vector([(x1-x0)/2+.0006,(y1-y0)/2+.0006,.0004]),
                          rgba=vector(p.get("rim_rgba",[.95,.015,.045,1])),contype="0",conaffinity="0")
            ET.SubElement(world,"geom",name=f"control_{p['id']}",type="box",pos=vector(pos),
                          size=vector([(x1-x0)/2,(y1-y0)/2,half_height]),rgba=".10 .11 .12 1",
                          solref=".003 1")
    return ET.tostring(root,encoding="unicode")


class DrumSim:
    def __init__(self,kit,duration_s=60,render_size=224):
        import mujoco
        self.mj=mujoco
        self.kit=kit
        self.module=robot_module(kit)
        self.model=mujoco.MjModel.from_xml_string(scene_xml(kit))
        self.data=mujoco.MjData(self.model)
        self.robot=self.module.RobotKinematics(self.model)
        self.renderer=None
        self.render_size=render_size
        self.rate=kit["audio_sample_rate"]
        self.duration_s=duration_s
        self.lookup=controls(kit)
        self.knobs=[p for p in kit["controls"] if p["kind"]=="knob"]
        self.knob_qindices=[self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f"knob_{p['id']}")] for p in self.knobs]
        knob_bodies={self.model.body(p["id"]).id for p in self.knobs}
        self.robot_geom_ids={i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] not in knob_bodies|{0}}
        self.samples={p["id"]:pad_sample(p,self.rate) for p in self.lookup.values() if p.get("kind")!="knob"}
        self.tip_ids={self.model.geom(name).id:arm for name,arm in self.module.TIP_GEOMS.items()}
        self.control_ids={mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,f"control_{p}"):p for p in self.lookup}
        self.finger_ids={self.model.geom(name).id:i for name,i in self.module.KNOB_FINGER_GEOMS.items()}
        self.reset()

    def reset(self):
        self.mj.mj_resetData(self.model,self.data)
        initial=self.module.rest_action(self.robot,self.kit)
        self.robot.set_positions(self.data,initial)
        self._set_control(initial)
        for i,k in enumerate(self.knobs):
            self.data.qpos[self.knob_qindices[i]]=k["initial_angle_rad"]
        self.mj.mj_forward(self.model,self.data)
        self.audio=np.zeros(round((self.duration_s+2)*self.rate),dtype=np.float32)
        self.events=[]; self.active=set(); self.bad_contacts=0; self.last_trigger={}
        self.release_since={}
        self.last_action=initial.copy()
        self.grasp_samples=[]; self.contact_force=np.zeros(6)
        self.bad_contact_pairs={}
        self.button_states={p["id"]:False for p in self.lookup.values() if p.get("kind")=="button"}
        return self.state()

    def state(self):
        return np.r_[self.robot.positions(self.data),self.robot.velocities(self.data)].astype(np.float32)

    def joint_positions(self):
        return self.robot.positions(self.data)

    def knob_angle(self,index):
        return float(self.data.qpos[self.knob_qindices[index]])

    def _set_control(self,action):
        self.data.ctrl[:self.module.ACTION_DIM]=action

    def render(self,camera="policy"):
        if self.renderer is None:
            self.renderer=self.mj.Renderer(self.model,height=self.render_size,width=self.render_size)
        self.renderer.update_scene(self.data,camera=camera)
        return self.renderer.render().copy()

    def step(self,action):
        action=np.asarray(action,dtype=float)
        low,high=self.module.action_bounds()
        limit=self.module.velocity_limits(self.kit)/self.kit["control_hz"]
        action=np.clip(self.last_action+np.clip(action-self.last_action,-limit,limit),low,high)
        self.last_action=action.copy()
        self._set_control(action)
        begin=len(self.events)
        for _ in range(10):
            self.mj.mj_step(self.model,self.data)
            contacts=set(); grasp={k["id"]:set() for k in self.knobs}
            for contact_index,contact in enumerate(self.data.contact):
                a,b=int(contact.geom1),int(contact.geom2)
                self.mj.mj_contactForce(self.model,self.data,contact_index,self.contact_force)
                intended=False
                for finger,control in ((a,b),(b,a)):
                    if control in self.control_ids:
                        name=self.control_ids[control]
                        is_knob=self.lookup[name].get("kind")=="knob"
                        if finger in self.finger_ids and is_knob:
                            intended=True
                            if self.contact_force[0]>.02: grasp[name].add(self.finger_ids[finger])
                        if finger in self.tip_ids and not is_knob: intended=True
                if not intended and (a in self.robot_geom_ids or b in self.robot_geom_ids) and (contact.dist < -.0005 or self.contact_force[0]>.5):
                    self.bad_contacts+=1
                    pair=" / ".join(self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_GEOM,g) or str(g) for g in (a,b))
                    self.bad_contact_pairs[pair]=self.bad_contact_pairs.get(pair,0)+1
                if b in self.tip_ids:
                    a,b=b,a
                if a in self.tip_ids and b in self.control_ids:
                    pad,arm=self.control_ids[b],self.tip_ids[a]
                    if self.lookup[pad].get("kind")=="knob":
                        continue
                    contacts.add((pad,arm))
            self.grasp_samples.append({"time_s":float(self.data.time),"fingers":{k:sorted(v) for k,v in grasp.items() if v},
                                       "angles_rad":{k["id"]:self.knob_angle(i) for i,k in enumerate(self.knobs)}})
            for pad,arm in contacts-self.active:
                # Electronic-pad debounce: one touch can create several solver contacts.
                if self.data.time-self.last_trigger.get((pad,arm),-1) < .08:
                    continue
                self.last_trigger[(pad,arm)]=float(self.data.time)
                kind="press" if self.lookup[pad].get("kind")=="button" else "hit"
                self.events.append({"time_s":float(self.data.time),"pad":pad,"arm":arm,"kind":kind})
                if kind=="press":
                    self.button_states[pad]=not self.button_states[pad]
                start=round(self.data.time*self.rate)
                sample=self.samples[pad]
                gain=1.
                for k,knob in enumerate(self.knobs):
                    if knob["affects_pad"]==pad:
                        gain*=float((self.knob_angle(k)+2.3562)/4.7124)
                count=min(len(sample),max(0,len(self.audio)-start))
                if count:
                    self.audio[start:start+count]+=sample[:count]*gain
            # A mesh can briefly lose individual solver contacts while its
            # fingertip still rests on the pad. Rearm after measured separation,
            # not a one-substep contact flicker or a landing bounce; no reference events are used.
            held=self.active|contacts
            for pad,arm in held-contacts:
                tip=next(g for g,a in self.tip_ids.items() if a==arm)
                control=self.model.geom("control_"+pad).id
                distance=self.mj.mj_geomDistance(self.model,self.data,tip,control,.01,None)
                if distance>.005:  # a landing bounce of a few mm is not a new hit
                    since=self.release_since.setdefault((pad,arm),float(self.data.time))
                    if self.data.time-since>=.02: held.remove((pad,arm))
                else:self.release_since.pop((pad,arm),None)
            for item in contacts:self.release_since.pop(item,None)
            self.active=held
        return self.state(),self.events[begin:]

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer=None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
