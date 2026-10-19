Open [the video player](index.html) for the latest SONIC turning trial. It has phase bookmarks, playback speed controls and 0.1-second stepping.

- [Focused video](turn-focus.mp4): support release through the shutdown tail.
- [Half-speed focused video](turn-focus-half-speed.mp4): the same frames played twice as slowly.
- [Full video](full-trial.mp4): includes initialization with the virtual support band.

The three views use fixed cameras. Cyan shows the robot's torso direction reconstructed from the saved pose. Orange shows the world-facing direction sent to the planner. Pink shows the active controller target, including its heading correction. White shows the final world-heading goal. The request was a relative turn of -15 degrees; the absolute goal was about -15.58 degrees because the robot began slightly off zero.

In the focused video, support release is around 1.0 s, standing reset 3.0 s, backward walking 6.2 s, the turn request 7.3 s, and interruption 15.4 s. The shutdown tail begins around 16.8 s, after the last recorded control tick. The exact shutdown time was not logged. The collapsing posture in that tail must be considered separately from the turn interruption.

These are offline renders of the recorded base and joint observations at about 10 Hz. The original trial did not save a screen recording. Frames hold the most recent pose at each playback timestamp; there is no pose interpolation or new physics rollout. The observations cannot resolve every 50 Hz foot or control event, and contact forces were not recorded. The current scene files match between the native worktree and original workspace, but the trial did not save an as-run scene hash.

This trial exercised SONIC locomotion and interruption handling. It did not query the bottle VLA checkpoint or attempt manipulation. The red floor sphere is a fixed model marker. The saved result reports an interrupted turn and a confirmed stationary command hold; that command acknowledgement alone does not establish physical stability.

[Frame timestamps and values](frame-map.csv) connect every full-video frame to its source pose row. [Replay metadata](replay-metadata.json) records the input hashes, joint mapping, rendering environment and exact phase boundaries. The [original evidence](../g1_harness_20261006_instrumented/README.md) is unchanged.

To recreate the videos on this workstation:

```bash
MUJOCO_GL=egl /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  docs/artifacts/g1_harness_20261006_replay/render_replay.py \
  --trial docs/artifacts/g1_harness_20261006_instrumented/native-negative-zero-run \
  --scene /home/jihun/work/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml \
  --output docs/artifacts/g1_harness_20261006_replay
```

The renderer loads MuJoCo directly and calls `mj_forward` to place the saved poses. It does not start DDS, a policy, a simulation loop or a hardware connection.
