# Trash subtask skills, 6 November 2026

The user chose the existing trash pick-and-bin subtask policy as the first
multi-skill set for VoLo on the G1. It knows nine prompts, the exact strings in
`pnp_trash_subtasks/meta/tasks.jsonl`:

| Step | Prompts |
|---|---|
| 1 | `approach brown table` |
| 2 | `pick the apple`, `pick the cup`, `pick the pill bottle`, `pick the pill box`, `pick the red bottle` |
| 3 | `turn left and approach the trash bin`, `turn right and approach the trash bin` |
| 4 | `put it in to the trash bin` |

Each of the 72 episodes is annotated with one prompt from each step; the
boundaries come from the annotation workbook (`annotation_provenance.json`).
[`configs/g1/trash-subtasks.example.yaml`](../../../configs/g1/trash-subtasks.example.yaml)
registers the nine skills. Steps 1–3 finish with `hold`, so the next skill
starts without a standing reset; step 4 resets standing.

## Checkpoints

Both runs were trained on 28 August 2026 on the H100 (`mncsvr11`) from
`nvidia/GR00T-N1.7-3B`, embodiment `UNITREE_G1_SONIC`, 20,000 steps, image
`jihun/gr00t-n1.7:626af89`.

| Run | Episodes | Checkpoint on the H100 |
|---|---:|---|
| all-72 | 72 (44 turn left, 28 turn right) | `/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_n17_20260828/train/pnp-trash-subtasks-gpu6-20260828/checkpoint-20000` |
| left-only | 44 (turn left) | `/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/train/pnp-trash-subtasks-left-only-gpu6-20260828/checkpoint-20000` |

The profile uses the all-72 checkpoint, copied to the workstation at
`/mnt/data/jihun/g1_models/pnp_trash_subtasks_n17_20260828/checkpoint-20000`.

**Compatibility with the native executor.** Its `final_processor_config.json`
and `embodiment_id.json` are byte-identical to those of the n17 bottle
checkpoint in `configs/g1/workstation.yaml`, which the harness has run: state
is legs, waist, arms, hands and projected gravity (43); actions are a 64-value
SONIC motion token plus 7+7 Dex3 joints over 40 steps. The trash data was
recorded with `policy/sonic_v1_1`, the SONIC models the current deploy runs, so
the tokens decode with the same decoder. Serve it with Isaac-GR00T's
PolicyServer on :15558, not the LeRobot server the handover loop uses.

## Open-loop evaluation

Neither run held out episodes, but the left-only run never saw the 28
turn-right episodes (17–44). Their approach, pick and put segments are unseen
examples of skills it was trained on. [`subtask_open_loop.py`](subtask_open_loop.py)
copies the image's `open_loop_eval.py` loop (16-step stride, recorded ego view,
state and prompt) and groups the per-frame action error by the prompt each chunk
was predicted under. Both checkpoints ran on the 28 turn-right episodes and on
10 turn-left training episodes (0, 4, 8, 12, 16, 45, 50, 55, 60, 65), GPU 7,
6 November. Raw rows: [left-only](open_loop_left_only.jsonl),
[all-72](open_loop_all72.jsonl).

Mean absolute error; the hand column is the worse of the two hands.

| Subtask | Episodes | left-only token | left-only hand | all-72 token | all-72 hand |
|---|---|---:|---:|---:|---:|
| approach brown table | unseen by left-only (28) | 0.0394 | 0.0012 | 0.0196 | 0.0018 |
| | training (10) | 0.0158 | 0.0009 | 0.0184 | 0.0018 |
| pick (all objects) | unseen by left-only (28) | 0.0475 | 0.0610 | 0.0144 | 0.0243 |
| | training (10) | 0.0124 | 0.0275 | 0.0145 | 0.0257 |
| put it in to the trash bin | unseen by left-only (28) | 0.0492 | 0.0796 | 0.0188 | 0.0203 |
| | training (10) | 0.0163 | 0.0208 | 0.0186 | 0.0202 |
| turn left and approach | training (10) | 0.0158 | 0.0023 | 0.0191 | 0.0021 |
| turn right and approach | unseen, untrained prompt (28) | 0.0489 | 0.0035 | 0.0191 | 0.0029 |

On episodes it never saw, the left-only checkpoint's token error is 2.5–3.8×
its training-episode error for every subtask, and its hand error for picking and
putting is 2.2–3.8×. The increase is about the same for every object (apple,
pill bottle, pill box, red bottle). Episodes 17–44 are one contiguous block, so
the gap may come from that session's layout as much as from unseen episodes;
these numbers cannot separate the two.

The all-72 checkpoint has no unseen episodes, so its rows are training error
only. Expect its error on a new scene to rise as the left-only one did.

Open-loop error replays recorded inputs. It does not show closed-loop
stability, task success in simulation or on the robot, or that the robot
stops where the next skill expects to start.

## Open before the robot

- **VLA walking is unbounded.** Two skills walk through motion tokens. The
  harness bounds only `walk_for`/`turn_by` (0.2 m/s, 5 s); a VLA skill can walk
  for up to `manipulation_deadline_s` (120 s). Needs a bound or a supervised
  stop plan before the first session.
- No simulation scene has the brown table, the objects and the bin.
- The completion criteria are written from the recordings' ego-view frames and
  have not been checked against a VLM on these recordings.
- The robot test needs the same table, objects and bin as the recordings.

## Reproduce

On the H100, with `R=/mnt/data01/jhkim/gr00t_runs` and `<run>` one of the
checkpoint directories above:

```bash
docker run --rm --gpus device=7 \
  -v /mnt/data01/huggingface:/root/.cache/huggingface \
  -v /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_subtasks:/dataset:ro \
  -v $R/pnp_trash_subtasks_n17_20260828/dataset_view:/outputs/dataset_view:ro \
  -v <run>:/ckpt/m:ro -v <out>:/out \
  --entrypoint /workspace/.venv/bin/python -w /workspace jihun/gr00t-n1.7:626af89 \
  /out/subtask_open_loop.py /ckpt/m /outputs/dataset_view /out/result.jsonl 17 18 ...
```

Mount the Hugging Face cache as the training containers did (read-write) and
leave offline mode off: the loader queries the Hub for the gated
`nvidia/Cosmos-Reason2-2B` and fails with `HF_HUB_OFFLINE=1`. A read-only
online mount was not tried.
