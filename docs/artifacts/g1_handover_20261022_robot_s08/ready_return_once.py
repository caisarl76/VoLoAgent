"""Stage 2 of the supervised s08 robot session: one production ready return through the harness RPC.

Usage (VoLoAgent venv, workstation):
    ready_return_once.py --endpoint ipc:///tmp/volo-pc2-harness.sock --profile session-profile.yaml OUT.json

Claims control with the session profile's registry digest, requests reset_ready, heartbeats every 0.2 s until the
native harness reports COMPLETED, INTERRUPTED or FAULT (or 30 s pass), then releases control. It records every
status. The operator keeps the deploy keyboard stop; this script never commands motion beyond the one reset_ready.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from vlm_orchestrator.harness.g1.client import G1Client
from vlm_orchestrator.harness.g1.registry import load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    profile = load_profile(a.profile)
    client = G1Client(a.endpoint)
    statuses, session = [], "stage2-ready-return"
    lease = None
    try:
        before = client.get_status()
        statuses.append(dict(at=time.monotonic(), **asdict(before)))
        lease = client.request("claim_control", {"registry_sha256": profile.registry_sha256}, session_id=session).result
        execution = client.request("reset_ready", {"execution_id": ""}, session_id=session,
                                   lease_id=lease.lease_id).result
        deadline, sequence = time.monotonic() + 30, 0
        while True:
            status = client.get_status()
            statuses.append(dict(at=time.monotonic(), **asdict(status)))
            if status.phase in ("COMPLETED", "INTERRUPTED", "FAULT") or time.monotonic() > deadline:
                break
            sequence += 1
            client.request("heartbeat", {"sequence": sequence}, session_id=session, lease_id=lease.lease_id)
            time.sleep(0.2)
        result = dict(execution_id=execution.execution_id, final_phase=status.phase, reason=status.reason,
                      timed_out=status.phase not in ("COMPLETED", "INTERRUPTED", "FAULT"))
    finally:
        if lease is not None:
            client.request("release_control", {}, session_id=session, lease_id=lease.lease_id)
        client.close()
    a.out.write_text(json.dumps(dict(result=result, statuses=statuses), indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
