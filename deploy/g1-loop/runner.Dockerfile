# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# G1 loop runner (vlm_orchestrator.harness.g1.cli) for PC2, aarch64, no GPU.
# Only what the CLI imports, pinned to the PC2 10-22 venv; the code runs from PYTHONPATH as on PC2.
FROM python:3.10-slim

RUN pip install --no-cache-dir pillow==12.2.0 numpy==1.26.4 openai==3.22.1 pyyaml==6.0.3 pyzmq==27.2.0

COPY vlm_orchestrator /opt/volo/vlm_orchestrator
ENV PYTHONPATH=/opt/volo
