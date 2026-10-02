# Recorded manipulation checks — 2026-10-02

Ten frames from the supplied apple and white-box episodes were evaluated with
`openai/gpt-5.6-sol` at `https://api.genon.ai/v1`. The evaluator uses the actual
G1 completion monitor and failure-handler prompt, with task-specific offline
criteria. It creates no robot client or action publisher. `GENON_API_KEY` was
read from the workstation's `.env`; credentials are absent from the artifacts.

The user labeled episode 11 successful apple placement and episode 0 a failed
box grasp followed by recovery. Individual frame labels are provisional session
annotations. A successful episode can end with a frame that leaves release
unclear. These annotations need review before becoming an acceptance dataset.

| Frame | Time | Provisional label | Model judgment |
| --- | ---: | --- | --- |
| Apple before grasp | 1.0 s | Incomplete | In progress |
| Apple held | 5.5 s | Incomplete | Unavailable: 10-second deadline |
| Apple above basket | 6.1 s | Incomplete | In progress: still held |
| Apple in basket | 7.2 s | Complete | In progress: release unclear |
| Box, wide view | 2.0 s | Incomplete | In progress |
| Box before grasp | 9.0 s | Incomplete | In progress |
| Box, first grasp attempt | 13.8 s | Incomplete | In progress |
| Box carried after recovery | 18.2 s | Incomplete | In progress |
| Box on plate | 21.2 s | Complete | Complete/next |
| Box on plate, later | 22.4 s | Complete | In progress: release unclear |

Nine responses arrived within the deadline. None of the six valid incomplete
frames was called complete. Two of three provisional positive labels disagreed
with the model. Valid response latency ranged from 2.292 to 3.979 seconds, with
a median of 2.688 seconds. The unavailable request is excluded from accuracy
counts, and retained as a latency failure.

One bounded [held-apple retry](../vision-genon-retry/result.json) returned
`in_progress` in 3.260 seconds. It agrees with the incomplete label; the original
timeout remains recorded. All seven distinct incomplete frames have a valid
non-complete judgment when that retry is included. This does not erase the
initial deadline failure.

Every case starts a separate monitor. A single model completion increments its
confirmation streak, so the recorded monitor outcome remains `in_progress` even
for the positive box frame. This is a frame-recognition check; it does not prove
two-frame mission confirmation or detection latency during moving manipulation.
At the runtime's one-second check interval, the apple's 7.28-second video also
has too little clearly completed tail for two stable post-release checks.

The historical case ID `box_out_of_view` refers to a wide view where the carton
and plate remain visible along the upper image edge. It is not a full-occlusion
example. Full object occlusion and unrecovered failure remain uncovered.

The model declined release confirmation when a hand remained close to the
object. Clear views after hand withdrawal, ideally another 3–5 seconds, will help
separate a conservative visual judgment from an actual missed success. Keep the
missed grasp and transport frames as negatives. A monitor diagnosis of failure
ends a live harness mission; the box recovery video does not enable automatic
retry or replanning.

Source video hashes, exact frame times, images, raw replies and decision times
are in [result.json](result.json). Criteria and labels are in
[vision_cases.json](../vision_cases.json).

```bash
.venv/bin/python docs/artifacts/g1_harness_20261001/evaluate_vision_cases.py \
  --vlm-model openai/gpt-5.6-sol \
  --vlm-base-url https://api.genon.ai/v1 \
  --env-file /home/jihun/work/VoLoAgent/.env \
  --output /tmp/g1-vision-evaluation
```

The endpoint rejected the SDK example's optional `thinking_token_budget`
parameter for this model. The successful evaluation uses the repository's
standard reasoning-model wrapper without that parameter. An initial sandbox
connection failure was retried with network access approved.

Apple and box criteria are offline evaluator inputs. The deployed registry still
contains only the verified bottle prompt and checkpoint.
