# Qwen handover monitor evaluation — 2026-10-13

`Qwen/Qwen3.6-27B` on H100 `.173` GPU 6 answered the recorded handover
checks faster than the saved Genon run. It rejected all four frames labeled as
valid offers and missed the initial slip frame. It is a candidate for further
evaluation; this run does not justify replacing the handover monitor.

The existing server was used without restarting it, downloading weights, or
changing its configuration. Its host endpoint binds to localhost on port
34002, so requests went through an SSH tunnel. GPU 6's engine process had a
parent serving the same model name. Selected process and API observations are
in [server-observations.json](server-observations.json); model revision and
quantization were not verified.

## Request behavior

All probes used the actual `HandoverMonitor`, its existing phase criteria,
chronological ego images, distinct-frame confirmation, persistent failure
latch, and the production 10-second decision deadline. No robot client was
created. The model adapter used temperature 0 and no SDK retries.

| Configuration, same bottle-on-desk case | Observed result |
| --- | --- |
| Default request, no generation cap | Monitor returned unavailable at 10.00 s; no final model reply was captured |
| Default thinking, 256-token cap | 5.43 s provider call; 256 completion tokens, finish reason `length`, reasoning present and final content absent; structured unavailable |
| Thinking disabled, 256-token cap | 1.13 s provider call; valid `yes`, 51 completion tokens, finish reason `stop` |

The uncapped timeout's provider thread had not finished when its report was
saved, so that report records only the start time. The capped probe confirms
that a short cap alone can be exhausted by reasoning before any JSON answer.
It does not prove that every default-mode request will time out.

The successful configuration sends:

```json
{
  "max_tokens": 256,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

`chat_template_kwargs` is sent in the OpenAI SDK's `extra_body`. The following
results all use this configuration. The runtime CLI's provider settings and
hardware profiles were not changed; it does not yet expose these evaluation
flags.

## Same 13 phase cases

| Measure | Qwen, this run | Genon GPT-5.6-sol, saved run |
| --- | --- | --- |
| Available decisions | 13/13 | 13/13 |
| Agreement with current review labels | 8/13 | 10/13 |
| Actual provider calls | 12 | 11 |
| Provider latency min / median / max | 1.12 / 1.67 / 3.77 s | 2.94 / 3.71 / 5.36 s |
| Offer frames labeled positive, returned yes | 0/4 | 3/4 |
| Initial slip at source 16/frame 350 | unknown | failure |
| Failure latch | Began at rescue frame 360 | Began at slip frame 350 |
| Yes on a negatively labeled frame in this set | 0 | 0 |

The call counts differ because the failure latch suppresses later provider
calls. Neither model's latched verdicts are additional model detections.
These timings compare a current local run with historical API evidence;
server load and cache state were not isolated.

| Case | Qwen frame outcome | Saved Genon outcome | Review accepts |
| --- | --- | --- | --- |
| desk-bottle | yes | yes | yes |
| closed-grasp | unknown | no | no / unknown |
| offer15-first | no | yes | yes |
| offer15-second | no | yes | yes |
| offer30-first | no | unknown | yes |
| offer30-second | no | yes | yes |
| person-contact | unknown | no | no / unknown |
| empty15-first | yes | unknown | yes |
| empty15-second | yes | unknown | yes |
| removal30-incomplete | unknown | unknown | no / unknown |
| slip16 | unknown | failure | failure |
| rescue16 | failure | latched failure | failure |
| later16 | latched failure | latched failure | failure |

Qwen's offer reasons confuse the hand with a closed gripper or the bottle with
an object still resting on the desk. One reason also requires an upright bottle;
the actual criterion permits a horizontally supported bottle. The labels and
prompts were kept unchanged rather than adjusted to make Qwen pass.

Qwen's initial slip response describes apparent slipping but returns unknown
because the final outcome is unclear. The system criterion requires visible
slipping to count as failure; that judgment arrives only in the rescue case.

Its isolated empty-hand sequence returns yes without a sampled intervening
taking frame. Saved Genon judgments were unknown for that missing context.
The label agreement count should therefore not be read as proof of successful
person removal.

## Four-frame person-taking sequence

Both models agreed with 3/4 conservative review labels and confirmed one unique
`wait_empty` phase. Qwen's median provider latency was 2.21 s; saved Genon's was
3.77 s. Both said yes at frame 405, whose review label remains no/unknown. This
boundary disagreement is unresolved, not a confirmed physical false-positive
event. Both then confirmed at frame 415 and remained affirmative at 425.

[comparison.json](comparison.json) recomputes unique phases from case IDs;
the historical Genon transfer summary counted two complete observations as
two phases. That original evidence is preserved unchanged.

## Evidence and limits

- [13-case Qwen decisions, reasons, usage and latency](phase-cases/result.json)
- [Qwen taking/empty-hand sequence](transfer-sequence/result.json)
- [Uncapped default probe](default-probe/result.json)
- [Capped thinking probe](capped-thinking-probe/result.json)
- [Fast probe](fast-probe/result.json)
- [Saved Genon phase baseline](../g1_handover_20261009_followup/vision-compatible/result.json)
- [Saved Genon temporal baseline](../g1_handover_20261009_followup/transfer-after/result.json)

`compare.py` verified identical case IDs, phases, timestamps, 10-second
deadlines, source-video hashes, and JPEG bytes against the baseline. It also
rehashed the three original videos. All images and reports remain recorded
timestamp checks: provider wall time is not continuous video replay. Source
15/30 are training references, not held-out success tests. Episode outcomes
come from the user; individual phase labels are investigator judgments.
This run measures no robot movement, table clearance, or closed-loop success.

The next useful check is Qwen-specific visual interpretation of open G1 hands
and bottle support on separately reviewed clips. Any prompt adjustment needs
validation on additional unseen cycles, including early rolling and person
occlusion, before a live-monitor change. Longer stable offerings remain needed
even with these faster calls.

## Reproduce

From the G1 harness worktree, open the tunnel in another terminal:

```bash
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.1:13402:127.0.0.1:34002 h100
```

The existing endpoint accepted a placeholder API key. Set an environment
variable for the evaluator, then run:

```bash
export QWEN_LOCAL_API_KEY=local
.venv/bin/python docs/artifacts/g1_handover_20261009_followup/evaluate_handover.py \
  --output /tmp/volo-qwen-phase-rerun \
  --model Qwen/Qwen3.6-27B --base-url http://127.0.0.1:13402/v1 \
  --api-key-env QWEN_LOCAL_API_KEY --disable-thinking --max-tokens 256
```

For the four-frame sequence, add:

```bash
--manifest docs/artifacts/g1_handover_20261009_followup/transfer-sequence.json
```

Exit code 1 indicates a label disagreement, including unavailable responses.
It does not mean the server or checker crashed. The evaluator's original
Genon model, URL and key-environment defaults remain unchanged.

Local verification used Python 3.13.13 and OpenAI SDK 3.22.1. Pytest plugin
autoload was disabled because the inherited ROS launch-testing plugin could
not import `lark`; no environment packages were installed or changed.
The focused evaluator, handover-monitor and CLI checks passed **27 tests**.
See [verification.json](verification.json) and [focused-tests.log](focused-tests.log).
The temporary SSH tunnel was closed after evaluation; the existing model
server was left running.
