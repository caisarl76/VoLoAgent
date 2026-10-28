# VLM size study for the loop's re-arm and taken checks (2026-10-27)

**Question.** The final system has two RTX 3060s: one for the policy, one for the VLM. Does a larger VLM than Qwen3-VL-2B make the camera checks reliable enough to run without an operator?

**Test set.** 169 real robot frames from sessions ready-A, ready-A-r24 and ready-A-r26 (ready pose A):
- 51 re-arm frames at the ready pose;
- 118 frames at the offer pose after a handover, 32 of them with the bottle still on the palm.

A vision agent labelled them (`labels.json`, 14 marked uncertain) and the controller spot-checked them. `eval_vlm.py` asks the production prompts (`rearm_monitor.PROMPT`, `EMPTY_PROMPT`, `TAKEN_PROMPT`). The re-arm check is scored on re-arm frames, the taken check on offer-pose frames.

| Model | Serving | Peak GPU (incl. ~1.7 GB desktop) | Re-arm false yes / missed (51) | **Taken: false "taken"** (32 on-palm) | Taken missed (86) | s/prompt (median) |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B bf16 | our transformers server | ~6.1 GB | 0 / 1 | 3 | 11 | 0.48 |
| Qwen3-VL-4B bf16 | our transformers server | 11.9 GB | 0 / 0 | 9 | 1 | 0.60 |
| Qwen3.5-9B AWQ-4bit (cyankiwi) | vLLM 0.32.0 docker, eager, image cap 640x480 | 11.5 GB | 0 / 0 | 18 | 2 | 0.83 |
| Qwen3.5-9B FP8-dynamic (RedHatAI) | vLLM 0.32.0 docker | does not fit: OOM while loading (10.1 GB of weights) | – | – | – | – |

Two other attempts on the taken check:
- a corner crop plus "bottle visible?" (4B): 3-6 false taken and 73-75 missed (`crop_ab.py`);
- two images, the hold's first frame as a reference plus the current one (9B AWQ): 14 false taken (`twoimg_ab.py`).

**Findings.**
- **Re-arm** works with every size. 4B and 9B remove 2B's single miss.
- **Taken** gets worse with size. Most false "taken" frames come from one hold (r24 trial 0): the bottle lies on the palm at the top-right edge of the head camera, partly outside the image and behind a person's sleeve. The larger models answer "not on hand" there consistently, while the labels (checked at full resolution) show it on the palm.
- The limit is the head camera's view of the offer pose, not model size.
- 2B and 4B share the same vision encoder (24 x 1024, ~0.41 B parameters). 8B/9B have larger ones (27 x 1152).
- Proprioception gives no signal either: the right-arm tracking error and the hand joints change by less than 0.04 rad when the bottle is lifted off (r26 telemetry).

**vLLM on a 3060 (Qwen3.5-9B AWQ).** It needs `--enforce-eager --max-num-batched-tokens 2048 --mm-processor-kwargs '{"max_pixels": 307200}' --gpu-memory-utilization 0.84`. Without them, the encoder profiling at the maximum image size runs out of memory.

## Follow-up: the taken check on r28-r31 holds (2026-10-27)

241 frames from the 12 holds of r28-r31 (`holds_r28_r31/items.json`). Labels: 50 with the bottle on the palm, 181 after the take, and 10 mid-grab frames excluded. The frames come from the evidence folders in each session, at their original paths. `eval_taken.py` asks `TAKEN_PROMPT` only.

| Model | False "taken" (50 on-palm) | Missed (181) | s/prompt | Peak GPU |
|---|---|---|---|---|
| Qwen3-VL-2B | 2 (r31 t0) | 48 | 0.42 | – |
| Qwen3-VL-4B | **0** | **0** | 0.53 | 8.8 GB (GPU1, no desktop) |
| Qwen3.5-9B AWQ | 0 | 19, plus many unparsed answers | 0.98 | 11.6 GB |

4B marks every take in these holds within one frame (~0.6 s), including r31 t1/t2, where the arm torque did not move. In the earlier set, its 9 false "taken" all come from r24 t0 (6 frames, bottle at the image edge behind a sleeve) and r26 (3 frames).

## Follow-up: lerobot reference sessions s17-s20 (2026-10-27, Qwen3-VL-4B on :8301)

Frames come from `/mnt/data/jihun/g1_eval_sessions/s1{7,8,9}_*/raw` and `s20_*/raw`, read only. Each episode was sampled at 1 Hz (551 frames) and labelled by hand:

| Label | Meaning | Frames |
|---|---|---|
| C | bottle upright on the case | 109 |
| G | robot hand at or grasping the bottle on the case | 150 |
| H | bottle on or in the robot hand | 164 |
| P | person holding the bottle | 15 |
| E | no bottle on the hand or case | 83 |
| U | unsure | 30 |

The loop's three prompts were asked on every frame (`lerobot_s17_s20/`).

- **Re-arm "bottle" yes:** C 109/109. False yes on G 41/150, mostly the reach frames before the fingers close, plus 0 of the H, P and E frames. The loop never runs re-arm during a reach.
- **Empty step yes:** E 80/83. False yes on C/G/H/P: 4/438.
- **Taken (`bottle_on_robot_hand` false) on H:** 14/164 at 1 Hz. Every one has the bottle at the top-right image edge, partly out of frame, several with a person's hand on it.
- **Dense check of the on-hand stretches at 0.6 s** (the loop's pace, `dense-4b.json`): 27/248 false. Runs of 3 or more in a row occur in 2 of 32 episodes:
  - s17 t5: 5 in a row at 7.6-9.6 s, a person's hand on the bottle at the offer;
  - s18 t9: 3 in a row at 7.6-8.6 s, as the arm swings into the offer.
- Most false runs come 0.6-1.6 s after the lift, while the arm moves. The loop's taken check runs only in the still offer-pose hold after the release; there, r28-r31 gave 0/50.

## Follow-up: a "hand visible" taken prompt (review P1, 2026-10-27)

The review proposed asking `{"robot_hand_visible", "bottle_on_robot_hand"}`, so that "taken" needs a visible empty hand. `taken_hand_visible_ab/` scores it against the current prompt with Qwen3-VL-4B.

**Set A**, the r28-r31 holds (50 bottle-on frames, 181 empty):

| Prompt | False "taken" | Empty-hand yes | Hand "not visible" on empty frames |
|---|---|---|---|
| Current | 0 | 181 | n/a |
| Visible version | 0 | 140 | 38 |

**Set B** (`setB.json`), the holds that decide R32 (22 bottle-on frames at the image edge, 72 empty):

| Prompt | False "taken" | Empty-hand yes |
|---|---|---|
| Current | 20 | 72 |
| Visible version | 4 | 21 |

- In the empty-hand holds, the visible version reports the hand as not visible in most frames.
- Three "empty" answers in a row then never occur in r34 t0 or r38 t4, and occur later in r36 t8.
- So R32 would have needed the operator's Enter in two of its four rescues. The current prompt's false "empty" answers at the edge were all blocked by R32's wrist condition (wrist yaw >= 0.67).
