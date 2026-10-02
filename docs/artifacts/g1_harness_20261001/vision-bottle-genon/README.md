# Bottle completion monitor — 2026-10-02

Six recorded-frame judgments agree with provisional labels: four unfinished
frames stay unfinished, and two distinct released-bottle frames confirm
completion in sequence. There are no unavailable replies or false-complete
claims in these four negative examples. The destination is now user-confirmed;
full carried-bottle occlusion and live-camera acceptance remain open.

The source is bottle-training episode 18 at
`/mnt/data/jihun/datasets/G1_WBT_GR00T/pnp_bottle_260916/videos/chunk-000/observation.images.ego_view/episode_000018.mp4`.
Its hash, timestamps, raw model replies, latency, image hashes, and monitor
outcomes are in [result.json](result.json). Frame images are in [frames](frames).
Source metadata marks retained demonstrations, without success/failure labels.
Direct session inspection supplies these frame annotations. The user confirmed
on 2026-10-02 that the green stool is the intended right-table destination;
see [label-review.json](label-review.json). Raw recorded results retain their
original pre-confirmation provenance, and model replies are unchanged.

| Frame time | Label | Model candidate | Monitor outcome |
| --- | --- | --- | --- |
| 1 s | Bottle/destination out of view | In progress | In progress |
| 12 s | Before grasp | In progress | In progress |
| 28 s | Held during transport | In progress | In progress |
| 31 s | Held above destination | In progress | In progress |
| 37 s | Released on green stool | Complete | In progress, streak 1 |
| 38 s | Released on green stool | Complete | Complete, streak 2 |

The last two frames share an execution, inference epoch, and monitor instance.
Their image hashes differ, and replay source time increases. The first positive
candidate cannot complete the execution by itself. The other cases are independent.
Calls use the real G1 monitor and `openai/gpt-5.6-sol` at
`https://api.genon.ai/v1`, with a 10-second deadline and no SDK retry. Latency
ranges from 2.667 to 3.775 seconds. No robot client or actuator publisher runs.

The exact trained prompt and registered completion criterion remain
`pick drink bottle and place it on the right table` and visible released
placement. These recordings add no new manipulation skill to the checkpoint.

Reproduce from the VoLoAgent worktree, using a **fresh** output directory:

```bash
.venv/bin/python docs/artifacts/g1_harness_20261001/evaluate_vision_cases.py \
  --manifest docs/artifacts/g1_harness_20261001/bottle_vision_cases.json \
  --output /tmp/g1-bottle-vision-fresh \
  --vlm-model openai/gpt-5.6-sol --vlm-base-url https://api.genon.ai/v1 \
  --env-file /home/jihun/work/VoLoAgent/.env
```

The key is read locally and omitted from the evidence. The 1-second frame is
an early floor view, not full occlusion of a bottle during manipulation. An
unrecovered failure is covered separately in the user-labeled
[episode-5 report](../vision-bottle-failure-genon/README.md); combined results
are in [bottle-monitor-validation.json](../bottle-monitor-validation.json).
This replay does not test a live camera, policy worker or complete bottle/reset
mission. The earlier apple/box deadline failure and unclear release judgments
remain in the acceptance record.
