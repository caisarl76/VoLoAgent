"""Can a local VLM on one RTX 3060 check 'a drink bottle standing upright on the black case, no hand at it'?

Usage (lerobot venv, transformers 5.x): CUDA_VISIBLE_DEVICES=0 upright_check.py MODEL_ID FRAMES_DIR OUT.json
Loads the model in bf16, asks one JSON question per real head-camera frame, and records the answers, per-frame
latency, load time and peak GPU memory. Labels are the by-eye labels in LABELS (2026-10-26).
"""
import json, sys, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# upright: a drink bottle stands upright on the black case; hand: a robot or human hand touches or holds that bottle.
LABELS = {
    "s08_t0_first": (True, False), "s08_t0_25": (True, False), "s08_t3_first": (True, False), "s08_t3_25": (True, False),
    "s16_t1_first": (True, False), "s16_t1_25": (True, True),
    "s16_t0_60": (False, True), "s16_t5_60": (False, True), "s13_t9_60": (False, True), "s13_t4_60": (False, False),
    "s08_t0_last": (False, False), "s08_t3_last": (False, False), "s16_t0_last": (False, False), "s16_t1_last": (False, False),
    "s16_t5_last": (False, False), "s13_t4_last": (False, False), "s13_t9_last": (False, False),
}
PROMPT = ("You see the head camera of a humanoid robot looking down at a black case. Answer only with JSON "
          '{"bottle_upright_on_case": true|false, "hand_at_bottle": true|false}. bottle_upright_on_case is true only if a '
          "drink bottle stands upright on top of the black case (not lying down, not held, not on the floor). "
          "hand_at_bottle is true if any robot or human hand touches or holds a bottle.")

model_id, frames, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
torch.cuda.reset_peak_memory_stats()
t = time.monotonic()
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0").eval()
load_s = time.monotonic() - t
rows = []
for name, (upright, hand) in LABELS.items():
    image = Image.open(frames / f"{name}.jpg").convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize(); t = time.monotonic()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    torch.cuda.synchronize(); latency = time.monotonic() - t
    text = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    try:
        answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        answer = {}
    rows.append(dict(frame=name, label_upright=upright, label_hand=hand, answer=answer, raw=text, latency_s=round(latency, 3),
                     upright_ok=answer.get("bottle_upright_on_case") == upright, hand_ok=answer.get("hand_at_bottle") == hand))
    print(name, "upright", upright, "->", answer.get("bottle_upright_on_case"), "| hand", hand, "->", answer.get("hand_at_bottle"), f"{latency:.2f}s", flush=True)
summary = dict(model=model_id, dtype="bfloat16", gpu=torch.cuda.get_device_name(0), load_s=round(load_s, 1),
               peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
               peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
               median_latency_s=sorted(r["latency_s"] for r in rows)[len(rows) // 2],
               upright_correct=f"{sum(r['upright_ok'] for r in rows)}/{len(rows)}", hand_correct=f"{sum(r['hand_ok'] for r in rows)}/{len(rows)}")
out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")
print(json.dumps(summary, indent=2))
