"""Generate an original 12-DoF MJCF from the shared specification."""

import xml.etree.ElementTree as ET
from .spec import LEGS, RobotSpec


def _v(values):
    return " ".join(f"{x:.10g}" for x in values)


def mjcf(spec: RobotSpec, *, scene=True):
    g = spec.geometry
    root = ET.Element("mujoco", model=spec.name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=str(spec.physics_dt_s), integrator="implicitfast",
                  solver="Newton", iterations="10", ls_iterations="20", cone="pyramidal")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.02")
    ET.SubElement(default, "geom", friction=f"{spec.floor_friction} 0.005 0.0001",
                  condim="3", contype="1", conaffinity="1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    world = ET.SubElement(root, "worldbody")
    if scene:
        ET.SubElement(world, "light", pos="0 -2 3", dir="0 0 -1")
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                      rgba="0.18 0.21 0.24 1")
    base = ET.SubElement(world, "body", name="base", pos=f"0 0 {spec.spawn_height_m}")
    ET.SubElement(base, "freejoint", name="root")
    ET.SubElement(base, "geom", name="base_collision", type="box",
                  size=_v([g.body_length_m / 2, g.body_width_m / 2, g.body_height_m / 2]),
                  mass=str(g.body_mass_kg), rgba="0.88 0.49 0.18 1")
    ET.SubElement(base, "site", name="imu", size="0.005")
    body_map = {"base": base}
    joints = {j.name: j for j in spec.joints}
    for leg in LEGS:
        sx, sy = (1 if leg[0] == "F" else -1), (1 if leg[1] == "L" else -1)
        hip = ET.SubElement(base, "body", name=f"{leg}_hip",
                            pos=_v([sx * g.body_length_m * .42, sy * g.body_width_m / 2, 0]))
        thigh = ET.SubElement(hip, "body", name=f"{leg}_thigh",
                              pos=_v([0, sy * g.hip_length_m, 0]))
        knee = ET.SubElement(thigh, "body", name=f"{leg}_knee",
                             pos=_v([0, 0, -g.upper_length_m]))
        for part, body, end, mass, axis in [
            ("hip", hip, [0, sy * g.hip_length_m, 0], g.hip_mass_kg, "1 0 0"),
            ("thigh", thigh, [0, 0, -g.upper_length_m], g.upper_mass_kg, "0 1 0"),
            ("knee", knee, [0, 0, -g.lower_length_m], g.lower_mass_kg, "0 1 0"),
        ]:
            name = f"{leg}_{part}"
            body_map[name] = body
            j = joints[name]
            ET.SubElement(body, "joint", name=name, axis=axis,
                          range=_v([j.lower_rad, j.upper_rad]),
                          armature=str(j.armature_kg_m2), frictionloss=str(j.frictionloss_nm))
            ET.SubElement(body, "geom", name=f"{name}_collision", type="capsule",
                          fromto=_v([0, 0, 0] + end), size=str(g.link_radius_m),
                          mass=str(mass), rgba="0.23 0.27 0.30 1")
        ET.SubElement(knee, "geom", name=f"{leg}_foot", type="sphere",
                      pos=_v([0, 0, -g.lower_length_m]), size=str(g.foot_radius_m),
                      mass=str(g.foot_mass_kg), rgba="0.10 0.12 0.14 1")
        ET.SubElement(knee, "site", name=f"{leg}_foot_site",
                      pos=_v([0, 0, -g.lower_length_m]), size="0.005")
    for name, inertia in spec.inertial_overrides.items():
        ET.SubElement(body_map[name], "inertial", mass=str(inertia.mass_kg),
                      pos=_v(inertia.com_m), diaginertia=_v(inertia.diagonal_kg_m2))
    actuators = ET.SubElement(root, "actuator")
    for j in spec.joints:
        ET.SubElement(actuators, "position", name=f"{j.name}_servo", joint=j.name,
                      kp=str(j.kp_nm_per_rad), kv=str(j.kd_nm_s_per_rad),
                      ctrlrange=_v([j.lower_rad, j.upper_rad]),
                      forcerange=_v([-j.effort_limit_nm, j.effort_limit_nm]))
    if scene:
        key = ET.SubElement(root, "keyframe")
        ET.SubElement(key, "key", name="home",
                      qpos=_v([0, 0, spec.spawn_height_m, 1, 0, 0, 0] +
                              [j.home_rad for j in spec.joints]),
                      ctrl=_v([j.home_rad for j in spec.joints]))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"
