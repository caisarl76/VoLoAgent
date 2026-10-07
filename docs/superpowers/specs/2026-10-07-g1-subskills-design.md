# G1 manipulation subskills

The user approved proceeding after the proposal to prepare independently callable
`pick bottle and hold` and `place held bottle on the stool` skills. The existing
checkpoint is trained on the whole bottle task. This extension prepares the
harness and review material; it does not establish that checkpoint's competence
at either new prompt.

The user subsequently confirmed that picking must stop above the source table,
immediately after lifting. A hold near the stool does not satisfy that skill.
The placement skill starts from the held pose above the source table.

## Behavior

A registered manipulation skill may choose `completion_action: hold` or
`completion_action: reset_standing`. Omission retains today's standing reset
with open hands. Hold completion pauses VLA, invalidates the action epoch, and
waits for fresh native planner feedback acknowledging the hold. It preserves
measured hand targets, releases ownership safely if the sequence ends there,
and allows a following manipulation to start without a standing reset.

The native executor must reject a PAUSED-to-manipulation transition until a
fresh hold is confirmed. If the old inference worker is still finishing, the
coordinator may retry an explicit NOT_READY response during this handoff for
at most the policy deadline. It must not retry ambiguous RPC failures. All
handoff checks require the same owner, runtime, execution and paused epoch.

Hold acknowledgement proves controller mode and telemetry freshness. It does
not prove contact force, bottle stability or manipulation competence. Vision
completion criteria still require visible evidence of the skill's result.

## Dataset preparation

Keep new prompts in a candidate catalogue outside the live workstation profile.
Prepare continuous review clips for source episodes 52 and 78 and retain source
episodes 5 and 18 as held-out evaluation material for a future checkpoint.
Do not turn sampled observations into accepted segment boundaries.

A review gate must validate source identity, whole-source split membership,
skill prompt, accepted review, success, boundary/terminal evidence, and segment
bounds before enumerating 40-frame windows. Every window stays inside its skill
segment. Training and held-out coverage are required for each candidate skill.
Failures and pending labels produce a structured rejection, including malformed
list/object values. A window manifest is a preparation artifact, not a GR00T
training dataset or authorization to launch training.

## Constraints

- Keep the active profile and checkpoint unchanged.
- Do not send hardware commands. Native IPC tests use a memory publisher.
- Preserve recorded evidence and original datasets; new material has a new path.
- Do not implement the deferred dashboard.
- Sign off commits with the actual author identity.

## Review focus

Check hand opening/reset ordering, fresh hold acknowledgement, stale epochs,
pending inference, lease loss, and whole-source train/evaluation separation.
Keep the distinction between candidate annotations and reviewed training data
explicit in reports and CLI output.
