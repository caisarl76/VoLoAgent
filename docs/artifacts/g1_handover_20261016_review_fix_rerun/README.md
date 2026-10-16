# Offer-trial rerun after the review fixes, 16 October 2026

This repeats the [online pelvis estimate](../g1_handover_20261015_pelvis_estimate/README.md)
offer trials at the measured desk on the review-fixed code.

**Both offers returned directly, 6/6, with no desk or bottle contact. The results
match the pre-fix runs.** No hardware ran, and both review flags remain false.

## Setup

- **Code:** native `6534d25`.
  - `30b2fdd`: feedforward gated by PLANNER mode and a 200 ms age limit; IEEE-bit
    non-finite check.
  - `6534d25`: heartbeat sequences and session retirement.
- **Binary:** `647b8f5f…`.
- **VoLoAgent:** `bdfb308`.
- **Simulation:** loopback, grounded physics, zero-speed planner, no VLA actions,
  GPU 1.
- **Desk:** 0.85 m top, 0.15 m edge.
- **Profile:** production example `configs/g1/handover.example.yaml`, with the
  feedforward off.
- **Probe:** `run_entry_probe.py --return profile`.

**Fixture change needed by RPC v1.** The acceptance probe (`ready_probe.py`)
and the entry probe's pre-pose keep-alive (`run_entry_probe.py`) now send
`{"sequence": time.monotonic_ns()}` as their heartbeat parameter. Both run in one
process and session, and the monotonic clock increases across them. The
executor now rejects the old empty heartbeat, so without this change the lease
would lapse. Older probes in other folders still send `{}`; they only reproduce
their own historical runs.

## Results

[`trials.json`](trials.json) holds per-run details, written by [`summarize.py`](summarize.py)
with the pelvis-estimate summary.

| Entry | Decision (logged) | Completed | Return time | Min conservative desk gap | Desk / bottle contacts | Pelvis estimate vs true | Preserved max error |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| ep15 offer | direct ×3 (hand above desk top) | 3/3 | 2.5–2.8 s | 0.8–10.4 mm | 0 / 0 | 0.787–0.788 vs 0.7859–0.7869 m | 0.062–0.082 rad |
| ep30 offer | direct ×3 (hand above desk top) | 3/3 | 2.8–3.6 s | 20.7–25.0 mm | 0 / 0 | 0.786–0.788 vs 0.7854–0.7870 m | 0.073–0.084 rad |

- **Pre-pose line:** each run's "raise first" line in `native.log` is the
  fixture pre-pose, not the return under test.
- **Leases:** the sequenced heartbeats kept the lease alive through every run.
- **Episode 15 clearance:** still millimetres, as before the fixes.

## Reproduce

From the VoLoAgent worktree, with GPU 1 checked for 2 GiB free:

```bash
env PYTHONPATH=. .venv/bin/python docs/artifacts/g1_handover_20261014_post_handover_entry/run_entry_probe.py --entry source15_f431 --return profile --gpu 1 --desk-front-m 0.15 --output /tmp/new-dir
/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native/.venv/bin/python docs/artifacts/g1_handover_20261016_review_fix_rerun/summarize.py
```
