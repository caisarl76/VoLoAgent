# G1 handover loop in Docker

Containers for the supervised loop in [the runbook](../../docs/instructions/g1-autonomous-loop-runbook.md). Every model is baked into its image at build time, so the running services need no internet, only the lab LAN.

| Service | Host | What | Baked weights |
|---|---|---|---|
| `vlm` (profile `workstation`, image `volo-g1-workstation`) | workstation, x86_64, GPU 0 | `scripts/g1_local_vlm_server.py` on :8301 | Qwen3-VL-4B-Instruct `ebb281ec` at `/models/qwen3-vl-4b` |
| `policy` (profile `workstation`, same image) | workstation, x86_64, GPU 1 | lerobot `examples/g1_dex3_training/sonic_policy_server.py` on :5571 | s08 `groot_handover261015_he_full` at its workstation path, plus `nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B` in `/hf` |
| `runner` (profile `pc2`) | PC2 `pc2_222`, aarch64 Jetson, no GPU | `vlm_orchestrator.harness.g1.cli`: `loop`, `verdict`, `status` | none |

VoLoAgent does not talk to DimOS; nothing here joins the DimOS compose.

- `policy` keeps the checkpoint at `/mnt/data/jihun/g1_models/wbt_groot_handover261015_he_full/pretrained_model` inside the image. The server reports that path as its model, and `loop_take_over` refuses unless it equals `checkpoint:` in the session profile.
- Both servers ran in one venv on the workstation, so they share one image. `policy-requirements.txt` is that venv's frozen package set (lerobot `feat/g1-combined-eval` `de54de313`). Build from that lerobot commit.
- The servers run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so a missing file fails at startup instead of reaching for the hub.

## Prerequisites (stay manual, outside these containers)

- **Native deploy (T1) and native harness (T2)** on PC2, started as in the runbook's startup order, including the one `k` to port 5580. The harness creates `/tmp/volo-g1-harness.sock`, which the runner reaches through the `/tmp` mount.
  - **Native and runner must match.** The RPC contract requires exact status keys, so this runner (VoLoAgent `work/g1-autonomous-loop-20261026`, R32 `take_hand_empty`) refuses an older native harness and the reverse. Use native GR00T-WholeBodyControl `f192d29` (same branch name), and the 2026-10-27 deploy build, which publishes `motor_torque`; without it every hold waits for Enter. `runner status` on PC2 shows a mismatch before a session.
  - The loop's thresholds and rulings (R24-R32) come from the runbook and the ledger, not from this directory. The compose file passes no loop flags.
- **The loop folder on PC2** (`~/GR00T-WholeBodyControl-loop-<date>`), with `session-profile.yaml` (`checkpoint_verified: true`) and `sessions/`, prepared as in "Before the session". Set `LOOP_DIR` to its absolute path.
- The telemetry recorder still runs from the native venv, as in the runbook.

## Build (online, before the demo)

Copy `env.example` to `.env` in this directory and fill in the block for the host. The build blocks name the weight sources on this workstation. Each host needs only its own block: a missing setting fails where it is used and names itself (for example `bind source path does not exist: /set-LOOP_DIR-in-.env`).

Workstation (one image, 44 GB of layers; `docker images` shows about 74 GB because it counts compressed and unpacked copies):

    docker compose --profile workstation build

PC2 (`unitree` is not in the `docker` group, so use `sudo`), natively on the Jetson:

    sudo docker compose --profile pc2 build runner

To move the workstation images to the demo workstation:

    docker save volo-g1-workstation | pigz -1 > /mnt/data/volo-g1-workstation.tar.gz
    # on the demo workstation
    gunzip -c volo-g1-workstation.tar.gz | docker load

Copy this directory with its `.env` along; `up` and `run` use the built images and do not rebuild.

Checked 2026-10-27 with `--network none`: the VLM loaded from the image and answered an image question, and the policy server
loaded, reported the profile's checkpoint path on `info` and returned a finite `[40, 78]` chunk (both on CPU, as the GPUs were in use).
The runner image passes `tests/test_g1_*.py`.

## Run (offline)

Workstation. Only one server can hold each port and GPU, so stop tmux servers on :8301 and :5571 first. The VLM refuses to load with less than `VLM_MIN_FREE_GB` free. The policy takes about 3-4 minutes to listen:

    docker compose --profile workstation up -d
    docker compose --profile workstation logs -f policy   # wait for "[server] ... ready on port 5571"
    curl -s http://<workstation>:8301/v1/models

The native harness asks the policy server for its model once, when its worker first connects: **restart T2 after any policy restart**.

PC2, one container per terminal, in tmux:

    sudo docker compose --profile pc2 run --rm runner status
    # T3
    sudo docker compose --profile pc2 run --rm runner loop --session-dir sessions/<name> --trials N
    # T4
    sudo docker compose --profile pc2 run --rm runner verdict --session-dir sessions/<name>

`loop` gets `--vlm-base-url`/`--vlm-model` from `VLM_BASE_URL`/`VLM_MODEL`; the `vlm` service serves under the same `VLM_MODEL` name (`--served-name`). Other loop flags go after the subcommand as usual. If the demo workstation has another address, set `VLM_BASE_URL` here and `policy_host` in the session profile.

- **Enter = stop works only in the attached terminal.** Do not run `loop` with `-d` or `up`; detaching (Ctrl-P Ctrl-Q) leaves the robot without its stop key. Ctrl-C in T3 is not a stop (runbook Recovery).
- The runner runs as uid 1000 because the harness socket is user-only.
- The loop folder is mounted at the same absolute path: `loop_trial` sends the session path to the native harness, which writes trial files there.
- The container shares the host's monotonic clock, so frame ages stay comparable to the harness.
