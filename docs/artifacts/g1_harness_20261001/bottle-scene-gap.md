# Bottle manipulation simulation gap — 2026-10-02

This is the earlier gap inspection. The subsequent
[sample scene and live trials](../g1_harness_20261002/README.md) resolve basic model
compatibility and run the real checkpoint, but do not demonstrate placement.

The current loopback SONIC scene cannot validate bottle placement: it has no
bottle, source table, or right-side destination. A legacy scene exists, but
requires adaptation before it can drive this harness and represent the trained
task. No end-to-end simulated or physical bottle mission was run.

Both XML models load with the workstation's MuJoCo installation. The compiled
model inspection is saved in [bottle-scene-inspection.json](bottle-scene-inspection.json).

| Property | SONIC default scene | Legacy bottle scene |
| --- | --- | --- |
| Path under GR00T-WholeBodyControl | `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml` | `decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml` |
| Position / velocity / actuator counts | 50 / 49 / 43 | 57 / 55 / 43 |
| Task geometry | None | One cylinder bottle and one front table |
| Ego camera | `head_camera`, attached to torso | `egoview`, attached to world |
| Right-hand actuator tail | Middle finger, then index finger | Index finger, then middle finger |

Matching actuator counts do not establish compatibility. The legacy scene's
right-hand ordering differs, its free bottle adds state entries, and its fixed
camera differs from the SONIC ego view. It also lacks the recorded task's
separate right-side green stool/table. Changing only `ROBOT_SCENE` would leave
these issues unresolved.

To close this gate, build a scene around the validated SONIC robot model and
joint/actuator order, add the source and destination surfaces and a suitable
bottle/contact model, and retain the ego-camera pipeline. Verify model mapping,
hand contact, and frame publication before running the selected checkpoint and
monitor together. Placement must be judged independently of policy serving and
planner-transition success. Supervised hardware evaluation remains a separate
route, after monitor acceptance and deployment readiness are established.
