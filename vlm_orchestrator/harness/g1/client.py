"""Thread-owned local RPC client with safe REQ retries."""

from __future__ import annotations

import threading
import uuid

import zmq

from .contract import (
    Request,
    Response,
    Status,
    ObservationSnapshot,
    decode_response,
    encode_request,
)


class RPCError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class G1Client:
    def __init__(self, endpoint: str, timeout_s: float = 0.5):
        if not endpoint.startswith("ipc:///") or not 0 < timeout_s <= 0.5:
            raise ValueError("Local IPC and timeout <=0.5 s required")
        self.endpoint, self.timeout_s = endpoint, timeout_s
        self.runtime_id = None
        self._thread = threading.get_ident()
        self._context = zmq.Context()
        self._socket = None

    def _reset_socket(self):
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, round(self.timeout_s * 1000))
        self._socket.setsockopt(zmq.RCVTIMEO, round(self.timeout_s * 1000))
        self._socket.setsockopt(zmq.MAXMSGSIZE, 16 * 1024 * 1024)
        self._socket.connect(self.endpoint)

    def request(
        self,
        method: str,
        params: dict[str, object],
        *,
        session_id: str | None = None,
        lease_id: str | None = None,
        request_id: str | None = None,
    ) -> Response:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Create a separate G1Client on each thread")
        if self.runtime_id is None and method != "get_status":
            self.get_status()
        rid = request_id or uuid.uuid4().hex
        wire = encode_request(
            Request(1, rid, self.runtime_id, session_id, lease_id, method, params)
        )
        for attempt in range(2):
            if self._socket is None:
                self._reset_socket()
            try:
                self._socket.send(wire)
                response = decode_response(self._socket.recv())
                if response.request_id != rid:
                    raise ValueError("Mismatched response request ID")
                if (
                    self.runtime_id is not None
                    and response.runtime_id != self.runtime_id
                ):
                    raise RPCError("STALE_RUNTIME", "Executor restarted")
                if response.error:
                    raise RPCError(response.error["code"], response.error["message"])
                self.runtime_id = response.runtime_id
                return response
            except zmq.Again as exc:
                self._reset_socket()
                if attempt:
                    raise TimeoutError("G1 executor RPC timeout") from exc
        raise AssertionError("Unreachable")

    def get_status(self) -> Status:
        result = self.request("get_status", {}).result
        if not isinstance(result, Status):
            raise ValueError("Expected Status")
        return result

    def observe(self) -> ObservationSnapshot:
        result = self.request("observe", {}).result
        if not isinstance(result, ObservationSnapshot):
            raise ValueError("Expected ObservationSnapshot")
        return result

    def close(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Close G1Client on its owning thread")
        if self._socket is not None:
            self._socket.close(linger=0)
        self._context.term()
