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
