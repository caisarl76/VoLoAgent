# G1 handover loop in Docker

Workstation containers for the supervised loop in [the runbook](../../docs/instructions/g1-autonomous-loop-runbook.md). Every model is baked into the image at build time, so the running servers need no internet, only the lab LAN. PC2 is not containerized.

| Service | Host | What | Baked weights |
|---|---|---|---|
| `vlm` (profile `workstation`, image `volo-g1-workstation`) | workstation, x86_64, GPU 0 | `scripts/g1_local_vlm_server.py` on :8301 | Qwen3-VL-4B-Instruct `ebb281ec` at `/models/qwen3-vl-4b` |
| `policy` (profile `workstation`, same image) | workstation, x86_64, GPU 1 | lerobot `examples/g1_dex3_training/sonic_policy_server.py` on :5571 | s08 `groot_handover261015_he_full` at its workstation path, plus `nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B` in `/hf` |

VoLoAgent does not talk to DimOS; nothing here joins the DimOS compose.

- `policy` keeps the checkpoint at `/mnt/data/jihun/g1_models/wbt_groot_handover261015_he_full/pretrained_model` inside the image. The server reports that path as its model, and `loop_take_over` refuses unless it equals `checkpoint:` in the session profile.
- Both servers ran in one venv on the workstation, so they share one image. `policy-requirements.txt` is that venv's frozen package set (lerobot `feat/g1-combined-eval` `de54de313`). Build from that lerobot commit.
- The servers run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so a missing file fails at startup instead of reaching for the hub.

## PC2 (native, as in the runbook)

PC2 runs everything natively, exactly as in the runbook: the deploy (T1), the native harness (T2), the loop runner (T3) and the verdict console (T4) from the loop folder `~/GR00T-WholeBodyControl-loop-<date>` with the harness venv, plus the telemetry recorder. Follow its "Before the session" and "Startup order"; this directory adds nothing there.

- **Native and runner must match.** The RPC contract requires exact status keys, so the runner (VoLoAgent `work/g1-autonomous-loop-20261026` `93c7afd`, R32 `take_hand_empty`) refuses an older native harness and the reverse. Use native GR00T-WholeBodyControl `f192d29` (same branch name) and the 2026-10-27 deploy build, which publishes `motor_torque`; without it every hold waits for Enter.
- The runner's `--vlm-model` must be this server's `VLM_MODEL` (`qwen3-vl-4b`). If the demo workstation has another address, change `--vlm-base-url` and `policy_host` in the session profile.
- The loop's thresholds and rulings (R24-R32) come from the runbook and the ledger, not from this directory.

## Build (online, before the demo)

Copy `env.example` to `.env` in this directory. The build block names the weight sources on this workstation; a missing one fails at its build context and names itself (for example `/set-LEROBOT_DIR-in-.env`). One image, 44 GB of layers; `docker images` shows about 74 GB because it counts compressed and unpacked copies:

    docker compose --profile workstation build

To move the image to the demo workstation:

    docker save volo-g1-workstation | pigz -1 > /mnt/data/volo-g1-workstation.tar.gz
    # on the demo workstation
    gunzip -c volo-g1-workstation.tar.gz | docker load

Copy this directory with its `.env` along; `up` uses the loaded image and does not rebuild.

Checked 2026-10-27 with `--network none`: the VLM loaded from the image and answered an image question, and the policy server
loaded, reported the profile's checkpoint path on `info` and returned a finite `[40, 78]` chunk, on CPU and again on GPU.

## Run (offline)

Workstation. Only one server can hold each port and GPU, so stop tmux servers on :8301 and :5571 first. The VLM refuses to load with less than `VLM_MIN_FREE_GB` free. The policy takes about 3-4 minutes to listen:

    docker compose --profile workstation up -d
    docker compose --profile workstation logs -f policy   # wait for "[server] ... ready on port 5571"
    curl -s http://<workstation>:8301/v1/models

The native harness asks the policy server for its model once, when its worker first connects: **restart T2 after any policy restart**.
