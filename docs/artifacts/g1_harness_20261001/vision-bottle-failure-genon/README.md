# Failed bottle episode — 2026-10-02

All four sampled frames remain incomplete. Genon returns `in_progress` for each,
with no completion candidate, confirmed completion, unavailable reply or label
disagreement. The user identified this episode as failed placement without
recovery. Frame annotations come from session inspection.

Source:
`/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_bottle_260916/videos/chunk-000/observation.images.ego_view/episode_000005.mp4`.
The clip lasts 19.54 seconds / 977 frames. Its SHA-256 is
`24d59c1358e8a95b7ad5b66880a879499b18183a72935cec2561612836d77445`.

| Time | Observed evidence | Monitor outcome |
| --- | --- | --- |
| 12 s | Bottle at the front source table, before grasp | In progress |
| 17 s | Grasp attempt; destination placement not visible | In progress |
| 18.8 s | Target no longer visibly identifiable near the left hand; other bottles remain on the source table | In progress |
| 19.4 s | Failed ending without visible intended placement | In progress |

The four frames share an execution and monitor, with increasing replay times.
Calls use `openai/gpt-5.6-sol` at `https://api.genon.ai/v1`, the exact trained
bottle prompt, a 10-second deadline, and no SDK retry. Replies take
2.988–4.967 seconds. [Raw results](result.json) preserve replies, image hashes,
latencies and monitor state; [frames](frames) preserve the inspected images.
No robot client or actuator publisher runs in this evaluation.

These samples validate absence of false success. The model did not classify
the episode as an explicit failure, so immediate failure detection remains
unvalidated. Target visibility loss after the grasp attempt is covered, but
the clip does not establish full obstruction of an already carried bottle.
The 18.8-second reasoning may refer to another bottle, so persistent target
identity also remains unvalidated.

Reproduce from the VoLoAgent worktree with a fresh output folder:

```bash
.venv/bin/python docs/artifacts/g1_harness_20261001/evaluate_vision_cases.py \
  --manifest docs/artifacts/g1_harness_20261001/bottle_failure_vision_cases.json \
  --output /tmp/g1-bottle-failure-vision-fresh \
  --vlm-model openai/gpt-5.6-sol --vlm-base-url https://api.genon.ai/v1 \
  --env-file /home/jihun/work/VoLoAgent/.env
```

Credentials are read locally and omitted from evidence. The
[combined bottle summary](../bottle-monitor-validation.json) includes this
failed episode and the [successful release](../vision-bottle-genon/README.md).
Live-camera and complete bottle/reset mission checks remain pending.
