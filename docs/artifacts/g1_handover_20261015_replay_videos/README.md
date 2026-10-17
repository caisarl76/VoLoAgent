# Ready-return replay videos, 15 October 2026

The clips replay saved physics from the simulation trials: each trial's 20 Hz
ground-truth `qpos` is rendered against its `scene.xml`. They are not new runs.
The camera views the robot from its right side; orange dots mark contacts.
They play in real time at 20 fps.

| Clip | Trial | What it shows |
| --- | --- | --- |
| [`1-old-raise-ep15.mp4`](1-old-raise-ep15.mp4) | `g1_handover_20261015_desk_edge_015/trials/e15-prod-01` | Measured desk, old rule (native `f356332`): raise first from the episode 15 offer, fingers land on the desk ~0.4 s in |
| [`2-direct-ep15.mp4`](2-direct-ep15.mp4) | `g1_handover_20261015_pelvis_estimate/trials/est-e15-01` | Measured desk, new rule (`ddd6008`): hand above desk top, direct return, no contact (millimetres of clearance) |
| [`3-direct-ep30.mp4`](3-direct-ep30.mp4) | `g1_handover_20261015_pelvis_estimate/trials/est-e30-01` | Measured desk, new rule: direct return from the episode 30 offer, 24–25 mm clearance |
| [`4-low-pelvis-ep30.mp4`](4-low-pelvis-ep30.mp4) | `g1_handover_20261015_pelvis_estimate/trials/low-e30-03` | Desk at 0.887 m (the 0.85 m desk seen from a 0.75 m pelvis): direct return from episode 30, one wrist-contact sample |
| [`5a-standing-no-ff.mp4`](5a-standing-no-ff.mp4) | `g1_handover_20261014_post_handover_entry/trials/merge-prod-standing-02` | 0.40 m edge, raise from grounded standing without feedforward: shoulder-yaw stall, `reset_timeout` |
| [`5b-standing-ff.mp4`](5b-standing-ff.mp4) | `g1_handover_20261015_yaw_feedforward/trials/ff-standing-01` | 0.40 m edge, raise from grounded standing with yaw gravity feedforward: completes |
| [`6-standing-spawn-at-desk.mp4`](6-standing-spawn-at-desk.mp4) | `g1_handover_20261015_desk_edge_015/trials/standing-prod-01` | Measured desk: the robot spawned standing at the desk puts its hands on the desktop and is pushed back |

The offer clips start 0.3 s before the desk restore, then cover the return
under test.

## Reproduce

[`render.py`](render.py) needs an OpenGL context; this machine used GLFW on the
local X display, since EGL failed here. From `docs/artifacts`, for example:

```bash
MUJOCO_GL=glfw DISPLAY=:0 /home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python \
  g1_handover_20261015_replay_videos/render.py g1_handover_20261015_pelvis_estimate/trials/est-e15-01 out.mp4 \
  "caption" '{"t0": 6.2}'
```

The options are `view` (`side` or `wide`), `t0` and `t1` (sim seconds), and
`still` (one PNG frame).
