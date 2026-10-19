"""Conservative bounds for collision screening; overlapping bounds are unknown."""
from itertools import product

import mujoco
import numpy as np


def geom_aabb(model,data,geom):
    kind=model.geom_type[geom]
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        mesh=model.geom_dataid[geom]
        vertices=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        vertices=np.array(list(product((-1,1),repeat=3)))*model.geom_size[geom]
    else:
        raise ValueError('Screen only supports mesh and box geometry')
    world=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
    assert np.isfinite(world).all()
    return world.min(0),world.max(0)


def aabb_gap(first,second):
    """Lower bound on separation; zero means bounds overlap, not proven contact."""
    low1,high1=first;low2,high2=second
    separation=np.maximum(np.maximum(low1-high2,low2-high1),0)
    return float(np.linalg.norm(separation))


def closest_bounds_gap(model,data,obstacles):
    nearest=dict(gap_m=float('inf'),actor=None,obstacle=None)
    actors=[]
    for g in range(model.ngeom):
        name=model.body(int(model.geom_bodyid[g])).name or ''
        if model.geom_contype[g] and name.startswith(('right_hand','right_wrist','right_elbow')):
            actors.append((g,name,geom_aabb(model,data,g)))
    assert actors
    for name in obstacles:
        bounds=geom_aabb(model,data,model.geom(name).id)
        for g,body,actor_bounds in actors:
            gap=aabb_gap(actor_bounds,bounds)
            if gap<nearest['gap_m']:nearest=dict(gap_m=gap,actor=body,obstacle=name)
    return nearest
