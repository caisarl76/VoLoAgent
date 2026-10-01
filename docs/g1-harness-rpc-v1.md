# G1 harness local RPC version 1

Status includes `locomotion_enabled: bool` (default false). Coordinators check
this capability before claiming a sequence containing walking or turning.

The G1 inference loop owns robot control. A background REP worker on `ipc:///tmp/volo-g1-harness.sock` accepts JSON requests and queues them for that loop. Each request returns promptly with acceptance/status; poll `get_status` to confirm a motion finished. The endpoint supports local IPC only and uses user-only permissions.

```json
{"version":1,"request_id":"unique-request","runtime_id":"executor-boot-id","session_id":"coordinator-id","lease_id":"issued-lease-id","method":"start_manipulation","params":{"skill_id":"bottle_to_right_table"}}
```

First call `get_status` with null runtime/session/lease IDs. Subsequent requests target its runtime ID. Claim takes a session ID and null lease ID; read-only calls need no lease. Motion and heartbeat calls require both session and lease IDs. Responses always contain `request_id`, `runtime_id`, `result`, and `error`; exactly one of result/error is non-null. Error has `code` and `message`.

| Method | Params | Result |
| --- | --- | --- |
| `get_status`, `observe` | `{}` | Status or RGB JPEG snapshot |
| `claim_control` | `registry_sha256` | Lease |
| `heartbeat`, `release_control` | `{}` | Status |
| `start_manipulation` | `skill_id` | Execution |
| `pause_manipulation`, `cancel` | `execution_id` | Execution |
| `reset_standing` | `execution_id`, `open_hands` | Execution; empty ID starts a standalone reset |
| `walk_for` | `direction`, `duration_s`, `speed_mps` | Execution; requires opt-in locomotion |
| `turn_by` | `angle_rad`, `rate_rps` | Execution; requires opt-in locomotion |

Unknown fields/methods, arbitrary prompt parameters, unsupported skill IDs, NaN/Infinity, and motion outside the profile limits fail before execution. Numeric strings and booleans in numeric fields are invalid. Standing reset preserves the heading measured at entry. Cancellation holds measured upper-body/hand targets and commands zero locomotion when fresh telemetry permits a planner handoff.

`registry_sha256` hashes the shared YAML bytes. `runtime_id` changes when the executor restarts. Retries reuse the exact session/request ID and payload; cached mutation replies cannot repeat motion or renew a lease. Different payloads with a reused ID fail with `REQUEST_CONFLICT`. Recreate a REQ socket after timeout. The coordinator heartbeat uses a separate thread/client/socket; it must keep renewing a 2 s lease every 0.25 s while vision runs. Claim does not start motors.

The exact dataclass fields are in `vlm_orchestrator/harness/g1/contract.py` and the design spec. `started_at`, `received_at`, and `expires_at` are workstation monotonic times. Camera frame IDs advance only on source timestamp changes; telemetry freshness requires advancing `index`. A sent mode command is a request. Fresh `planner_reference_active=true` confirms planner activation; false establishes non-planner only. `controller_running` is the operator-controlled runner flag. Checkpoint identity is configured/expected until independently verified.

Errors: `INVALID_REQUEST`, `PROFILE_MISMATCH`, `BUSY`, `NOT_READY`, `NOT_OWNER`, `EXPIRED_LEASE`, `STALE_RUNTIME`, `REQUEST_CONFLICT`, `UNAVAILABLE`, `UNSUPPORTED_SKILL`, `FAULT`. Invalid requests may have an empty echoed request ID when no valid envelope could be decoded.

The native repository carries data-only mirrors `harness_contract.py` and `harness_profile.py`, plus the same golden fixture. Integration tests compare them, allowing only the native relative-import rename. Neither environment imports the other repository's package.
