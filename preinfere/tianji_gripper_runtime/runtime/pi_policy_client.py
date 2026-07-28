"""Websocket client wrapper for openpi / pi0.5 policy servers."""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from . import schema


class PiPolicyServerError(RuntimeError):
    """Raised when the pi policy server is unavailable or returns bad data."""


class PiPolicyDependencyError(PiPolicyServerError):
    """Raised when the local Python environment cannot create a pi client."""


class PiPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_ms: int = 15000,
        action_layout: str = "ziyi_15d_right_left_right_gripper",
        control_mode: str = "right_arm_right_gripper",
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.action_layout = action_layout
        self.control_mode = control_mode
        self.last_latency_ms: float | None = None
        try:
            import websockets.sync.client

            from openpi_client import msgpack_numpy
        except Exception as exc:  # noqa: BLE001
            raise PiPolicyDependencyError(
                "pi policy client dependencies are unavailable in "
                f"{sys.executable}: {type(exc).__name__}: {exc}. "
                "Run this entrypoint with `uv run python` from the openpi repository."
            ) from exc
        self._websockets_client = websockets.sync.client
        self._packer = msgpack_numpy.Packer()
        self._msgpack_numpy = msgpack_numpy
        self._ws = None
        self._server_metadata: dict[str, Any] = {}
        self._connect()

    @property
    def policy_state_dim(self) -> int:
        if self.action_layout == "ziyi_15d_right_left_right_gripper":
            return 15
        if self.action_layout == "ziyi_16d_right_left_dual_gripper":
            return 16
        raise PiPolicyServerError(f"unsupported pi action layout: {self.action_layout!r}")

    def _connect(self) -> None:
        uri = self.host if self.host.startswith("ws") else f"ws://{self.host}"
        if self.port is not None:
            uri += f":{self.port}"
        try:
            self._ws = self._websockets_client.connect(
                uri,
                compression=None,
                max_size=None,
                open_timeout=max(float(self.timeout_ms) / 1000.0, 1.0),
                # First pi0.5 inference can spend a long time compiling. During
                # that window the server may not answer websocket pings, so the
                # default keepalive can close an otherwise healthy connection.
                ping_interval=None,
            )
            metadata = self._ws.recv()
            if isinstance(metadata, str):
                raise PiPolicyServerError(f"pi server returned text during handshake: {metadata}")
            self._server_metadata = self._msgpack_numpy.unpackb(metadata)
        except Exception as exc:  # noqa: BLE001
            self._ws = None
            raise PiPolicyServerError(f"failed to connect pi policy server {uri}: {exc}") from exc

    def ping(self) -> bool:
        return self._ws is not None

    def reset(self) -> None:
        return None

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "video": _LocalModalityConfig(
                modality_keys=["head", "left_wrist", "right_wrist"],
                delta_indices=[0],
            ),
            "state": _LocalModalityConfig(
                modality_keys=[f"state_{self.policy_state_dim}"],
                delta_indices=[0],
            ),
            "language": _LocalModalityConfig(modality_keys=["prompt"], delta_indices=[0]),
        }

    def predict_action_chunk(
        self,
        observation: dict[str, Any],
        *,
        min_horizon: int = 1,
    ) -> np.ndarray:
        if self._ws is None:
            raise PiPolicyServerError("pi policy server is not connected")
        start = time.perf_counter()
        try:
            self._ws.send(self._packer.pack(observation))
            response = self._ws.recv()
        except Exception as exc:  # noqa: BLE001
            raise PiPolicyServerError(f"pi policy inference failed: {exc}") from exc
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(response, str):
            raise PiPolicyServerError(f"pi policy server returned error:\n{response}")
        data = self._msgpack_numpy.unpackb(response)
        if not isinstance(data, dict) or "actions" not in data:
            raise PiPolicyServerError(f"pi policy response must contain 'actions', got {type(data)!r}")
        action_chunk = self._extract_action_chunk(np.asarray(data["actions"], dtype=np.float32))
        if action_chunk.ndim != 2:
            raise PiPolicyServerError(f"action chunk must be 2-D [H,8], got {action_chunk.shape}")
        expected_dim = schema.action_dim_for_mode(self.control_mode)
        if action_chunk.shape[1] != expected_dim:
            raise PiPolicyServerError(
                f"action chunk width must be {expected_dim}, got {action_chunk.shape[1]}"
            )
        if action_chunk.shape[0] < min_horizon:
            raise PiPolicyServerError(
                f"action chunk horizon {action_chunk.shape[0]} < requested {min_horizon}"
            )
        if not np.all(np.isfinite(action_chunk)):
            raise PiPolicyServerError("action chunk contains NaN or Inf")
        return action_chunk.astype(np.float32, copy=False)

    def _extract_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise PiPolicyServerError(f"pi actions must have shape [H,D], got {arr.shape}")
        if self.control_mode == "dual_arm_dual_gripper":
            if self.action_layout != "ziyi_16d_right_left_dual_gripper":
                raise PiPolicyServerError(
                    "dual-arm dual-gripper control requires "
                    f"'ziyi_16d_right_left_dual_gripper', got {self.action_layout!r}"
                )
            if arr.shape[1] < 16:
                raise PiPolicyServerError(
                    f"dual-arm dual-gripper layout needs at least 16 dims, got {arr.shape[1]}"
                )
            return np.concatenate(
                [
                    arr[:, 0:7],
                    arr[:, 7:14],
                    arr[:, 14:15],
                    arr[:, 15:16],
                ],
                axis=1,
            )
        if self.action_layout == "ziyi_15d_right_left_right_gripper":
            if arr.shape[1] < 15:
                raise PiPolicyServerError(
                    f"ziyi pi 15D action layout needs at least 15 dims, got {arr.shape[1]}"
                )
            return np.concatenate([arr[:, 0:7], arr[:, 14:15]], axis=1)
        if self.action_layout == "ziyi_16d_right_left_dual_gripper":
            if arr.shape[1] < 16:
                raise PiPolicyServerError(
                    f"ziyi pi 16D action layout needs at least 16 dims, got {arr.shape[1]}"
                )
            return np.concatenate([arr[:, 0:7], arr[:, 14:15]], axis=1)
        raise PiPolicyServerError(f"unsupported pi action layout: {self.action_layout!r}")


class _LocalModalityConfig:
    def __init__(self, *, modality_keys: list[str], delta_indices: list[int]) -> None:
        self.modality_keys = modality_keys
        self.delta_indices = delta_indices
