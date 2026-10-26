"""The real handover setup in the acceptance scene: black case, PET bottle and the fitted D435 head camera."""
import xml.etree.ElementTree as ET

import numpy as np

CASE_M = (0.475, 0.35)  # Black flight case, depth x width (lerobot isaac_scene.CASE, measured 2026-10-19).
BOTTLE_RADIUS_M, BOTTLE_HALF_HEIGHT_M = 0.033, 0.0775  # Clear PET body, 15.5 cm; the 1.7 cm cap is visual only.
# Where bottles stood in the real head frames, pelvis heading frame (lerobot Isaac task: 0.33, -0.065 +- 0.04).
BOTTLE_PELVIS_M = (0.33, -0.065)


def vector(values):
    return " ".join(f"{v:.6g}" for v in values)


def restyle_scene(path, box_front_m, box_height_m, camera):
    """The real case, bottle and head camera, on the acceptance scene's geometry and names."""
    tree = ET.parse(path)
    world = tree.getroot().find("worldbody")
    depth, width = CASE_M
    geoms = {g.get("name"): g for g in world.iter("geom")}
    geoms["desk_top"].set("pos", vector([box_front_m + depth / 2, 0, box_height_m / 2]))
    geoms["desk_top"].set("size", vector([depth / 2, width / 2, box_height_m / 2]))
    geoms["desk_top"].set("rgba", "0.03 0.03 0.03 1")
    for i in range(4):  # The legs sit inside the solid case (kept: the worker moves them with it).
        geoms[f"desk_leg_{i}"].set("pos", vector([box_front_m + depth / 2, 0, box_height_m / 4]))
        geoms[f"desk_leg_{i}"].set("size", "0.01 0.01 0.01")
        geoms[f"desk_leg_{i}"].set("rgba", "0.03 0.03 0.03 1")
    geoms["task_bottle_geom"].set("size", vector([BOTTLE_RADIUS_M, BOTTLE_HALF_HEIGHT_M]))
    geoms["task_bottle_geom"].set("rgba", "0.72 0.76 0.78 1")
    bottle = next(b for b in world.iter("body") if b.get("name") == "task_bottle")
    ET.SubElement(bottle, "geom", name="task_bottle_cap", type="cylinder", size="0.0165 0.0085",
                  pos=vector([0, 0, BOTTLE_HALF_HEIGHT_M + 0.0085]), rgba="0.78 0.86 0.12 1",
                  contype="0", conaffinity="0", density="0")
    torso = next(b for b in world.iter("body") if b.get("name") == "torso_link")
    w, x, y, z = camera["rot_wxyz"]  # World convention: the camera looks along +x, up +z.
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    right, up = rotation @ [0, -1, 0], rotation @ [0, 0, 1]  # MuJoCo cameras look along -z, x right, y up.
    fovy = np.degrees(2 * np.arctan(camera["height"] / 2 / camera["fy"]))
    ET.SubElement(torso, "camera", name="d435_head", pos=vector(camera["pos"]), xyaxes=vector([*right, *up]),
                  fovy=f"{fovy:.4f}")
    tree.write(path)
    return dict(case_m=list(CASE_M), fovy_deg=float(fovy), xyaxes=[float(v) for v in [*right, *up]])
