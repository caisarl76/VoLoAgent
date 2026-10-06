"""Independent simulation placement evidence; never controls the robot."""

import math


def sample_bottle_truth(model, data, *, now):
    """Measure this scene's cylindrical bottle and every colliding robot link."""
    import mujoco
    import numpy as np

    geom = model.geom("task_bottle_geom").id
    body = model.geom_bodyid[geom]
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER
    pelvis = model.body("pelvis").id
    axis_z = abs(float(data.geom_xmat[geom].reshape(3, 3)[2, 2]))
    tilt = math.sqrt(max(0.0, 1-axis_z**2))
    radius, half_height = model.geom_size[geom, :2]
    vertical_extent = half_height*axis_z+radius*tilt
    horizontal_extent = half_height*tilt+radius
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    contacts = []
    for contact in data.contact:
        if geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == geom else contact.geom1
        ancestor = int(model.geom_bodyid[other])
        is_robot = False
        while ancestor:
            if ancestor == pelvis:
                is_robot = True
                break
            ancestor = int(model.body_parentid[ancestor])
        contacts.append({"geom": model.geom(other).name or model.body(model.geom_bodyid[other]).name,
                         "is_robot": is_robot, "position_z": float(contact.pos[2]),
                         "normal_abs_z": abs(float(contact.frame[2]))})
    return {"at": now, "sim_time": float(data.time),
            "bottle_position": data.geom_xpos[geom].tolist(),
            "bottle_bottom_z": float(data.geom_xpos[geom, 2]-vertical_extent),
            "bottle_xy_radius": float(horizontal_extent),
            "bottle_linear_speed": float(np.linalg.norm(velocity[3:])),
            "bottle_angular_speed": float(np.linalg.norm(velocity[:3])), "contacts": contacts}


def placement_confirmed(samples, destination, *, now):
    """Require 0.5s of fresh, stable top support without any robot contact.

    Geometry bounds come from the simulated bottle's oriented collision shape.
    This validates simulated object truth, not perception or scene calibration.
    Older recordings without the required fields cannot establish placement.
    """
    try:
        if not samples or not math.isfinite(now):
            return False
        end = float(samples[-1]["at"])
        if not 0 <= now-end <= 0.3:
            return False
        recent = []
        for sample in reversed(samples):
            recent.append(sample)
            if end-float(sample["at"]) >= 0.5-1e-9:
                break
        recent.reverse()
        if len(recent) < 4 or end-float(recent[0]["at"]) < 0.5-1e-9:
            return False
        previous = None
        for row in recent:
            at, sim_time = float(row["at"]), float(row["sim_time"])
            xyz = row["bottle_position"]
            bottom = float(row["bottle_bottom_z"])
            radius = float(row["bottle_xy_radius"])
            linear, angular = float(row["bottle_linear_speed"]), float(row["bottle_angular_speed"])
            if len(xyz) != 3 or not all(math.isfinite(v) for v in [at, sim_time, *xyz, bottom, radius, linear, angular]):
                return False
            if previous and not (0 < at-previous[0] <= 0.2+1e-9 and sim_time > previous[1]):
                return False
            previous = (at, sim_time)
            if not (0 <= linear <= 0.03 and 0 <= angular <= 0.1 and radius > 0):
                return False
            if math.hypot(xyz[0]-destination["xy"][0], xyz[1]-destination["xy"][1])+radius > destination["radius"]:
                return False
            if abs(bottom-destination["height"]) > 0.01:
                return False
            contacts = row["contacts"]
            if not contacts:
                return False
            for contact in contacts:
                z, vertical = float(contact["position_z"]), float(contact["normal_abs_z"])
                if (contact["geom"] != destination["geom"] or contact["is_robot"] is not False
                        or not math.isfinite(z) or not math.isfinite(vertical)
                        or abs(z-destination["height"]) > 0.01 or not 0.9 <= vertical <= 1.0+1e-9):
                    return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
