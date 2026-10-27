# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Workstation image (x86_64 + CUDA) for both G1 loop servers, which run in one venv on the workstation:
#   policy: lerobot examples/g1_dex3_training/sonic_policy_server.py (s08 GR00T)
#   vlm:    VoLoAgent scripts/g1_local_vlm_server.py (Qwen3-VL-4B)
# Build context: this directory; `lerobot`, `volo_scripts` and the weight dirs come in as named contexts.
# Every weight file is baked in; the servers run with no internet.
FROM python:3.13-slim

# opencv-python (not headless) is in the venv; evdev builds from source against the kernel headers
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 gcc libc6-dev linux-libc-dev \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.14

# The exact package set of the venv both servers run in.
RUN --mount=type=bind,source=policy-requirements.txt,target=/tmp/req.txt \
    uv pip install --system --no-cache -r /tmp/req.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
# only what the policy server needs, which keeps the checkout's .venv and data out of the transfer
COPY --from=lerobot pyproject.toml setup.py README.md /opt/lerobot/
COPY --from=lerobot src /opt/lerobot/src
COPY --from=lerobot examples/g1_dex3_training /opt/lerobot/examples/g1_dex3_training
RUN uv pip install --system --no-cache --no-deps -e /opt/lerobot

# Same path as on the workstation: the server reports it as its model, and loop_take_over
# refuses unless it equals the session profile's checkpoint.
ARG POLICY_PATH=/mnt/data/jihun/g1_models/wbt_groot_handover261015_he_full/pretrained_model
RUN --mount=type=bind,from=policy_ckpt,target=/src mkdir -p "$POLICY_PATH" && cp -rL /src/. "$POLICY_PATH"/
# base_model_path and the preprocessor's model_name are hub IDs, resolved offline from this cache
RUN --mount=type=bind,from=hub_gr00t,target=/src mkdir -p /hf/hub/models--nvidia--GR00T-N1.7-3B && cp -a /src/. /hf/hub/models--nvidia--GR00T-N1.7-3B/
RUN --mount=type=bind,from=hub_cosmos,target=/src mkdir -p /hf/hub/models--nvidia--Cosmos-Reason2-2B && cp -a /src/. /hf/hub/models--nvidia--Cosmos-Reason2-2B/
# snapshot files are symlinks into ../../blobs: dereference into a plain directory
ARG VLM_REVISION
RUN --mount=type=bind,from=vlm_hub,target=/src mkdir -p /models && cp -rL "/src/snapshots/$VLM_REVISION" /models/qwen3-vl-4b

# last, so a server change does not re-copy the weights
COPY --from=volo_scripts g1_local_vlm_server.py /opt/volo/g1_local_vlm_server.py

ENV HF_HOME=/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 POLICY_PATH=$POLICY_PATH
WORKDIR /opt/lerobot/examples/g1_dex3_training
