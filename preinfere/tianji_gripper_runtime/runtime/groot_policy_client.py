"""Dependency-light GR00T PolicyClient wrapper for 8-DoF gripper checkpoints."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from . import schema


class PolicyServerError(RuntimeError):
    """Raised when the GR00T policy server is unavailable or returns bad data."""


class RemoteModalityConfig:
    __slots__ = (
        "delta_indices",
        "modality_keys",
        "sin_cos_embedding_keys",
        "mean_std_embedding_keys",
        "action_configs",
    )

    def __init__(self, **kwargs: Any) -> None:
        self.delta_indices = list(kwargs.get("delta_indices", []) or [])
        self.modality_keys = list(kwargs.get("modality_keys", []) or [])
        self.sin_cos_embedding_keys = kwargs.get("sin_cos_embedding_keys")
        self.mean_std_embedding_keys = kwargs.get("mean_std_embedding_keys")
        self.action_configs = kwargs.get("action_configs")


class _VendoredPolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        try:
            import msgpack
            import msgpack_numpy as mnp
            import zmq
        except Exception as exc:  # noqa: BLE001
            raise PolicyServerError(
                "vendored policy client needs pyzmq, msgpack, msgpack-numpy"
            ) from exc
        self._zmq = zmq
        self._msgpack = msgpack
        self._mnp = mnp
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._context = zmq.Context.instance()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self._context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _to_bytes(self, data: Any) -> bytes:
        return self._msgpack.packb(data, default=self._mnp.encode)

    def _from_bytes(self, data: bytes) -> Any:
        return self._msgpack.unpackb(data, object_hook=self._object_hook, raw=False)

    def _object_hook(self, obj: Any) -> Any:
        return self._mnp.decode(obj, chain=_decode_modality_config)

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.socket.send(self._to_bytes(request))
            message = self.socket.recv()
        except self._zmq.error.Again:
            self.socket.close(linger=0)
            self._init_socket()
            raise
        response = self._from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except self._zmq.error.ZMQError:
            self.socket.close(linger=0)
            self._init_socket()
            return False

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        return tuple(response)

    def reset(self, options: dict[str, Any] | None = None) -> Any:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)


def _decode_modality_config(obj: Any) -> Any:
    if isinstance(obj, dict) and ("__ModalityConfig__" in obj or b"__ModalityConfig__" in obj):
        key = "as_json" if "as_json" in obj else b"as_json"
        as_json = obj[key]
        kwargs = {
            (k.decode() if isinstance(k, bytes) else k): v for k, v in as_json.items()
        }
        return RemoteModalityConfig(**kwargs)
    return obj


def _force_vendored() -> bool:
    raw = os.environ.get("GROOT_FORCE_VENDORED_POLICY_CLIENT", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _make_client(host: str, port: int, timeout_ms: int):
    if not _force_vendored():
        try:
            from gr00t.policy.server_client import PolicyClient

            return PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)
        except Exception:
            pass
    return _VendoredPolicyClient(host, port, timeout_ms)


class GrootPolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 15000) -> None:
        self._client = _make_client(host, port, timeout_ms)
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.last_latency_ms: float | None = None
        self._modality_configs: dict[str, Any] | None = None

    def ping(self) -> bool:
        return bool(self._client.ping())

    def reset(self) -> None:
        self._client.reset(options=None)

    def get_modality_config(self) -> dict[str, Any]:
        if self._modality_configs is None:
            try:
                self._modality_configs = self._client.get_modality_config()
            except Exception as exc:  # noqa: BLE001
                raise PolicyServerError(f"failed to query modality config: {exc}") from exc
        return self._modality_configs

    def predict_action_chunk(
        self,
        observation: dict[str, Any],
        *,
        min_horizon: int = 1,
    ) -> np.ndarray:
        start = time.perf_counter()
        try:
            action, _info = self._client.get_action(observation)
        except Exception as exc:  # noqa: BLE001
            raise PolicyServerError(f"policy server get_action failed: {exc}") from exc
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0

        action_chunk = self._flatten_action_dict(action)
        if action_chunk.ndim != 2:
            raise PolicyServerError(f"action chunk must be 2-D [H,8], got {action_chunk.shape}")
        if action_chunk.shape[1] != schema.ACTION_DIM:
            raise PolicyServerError(
                f"action chunk width must be {schema.ACTION_DIM}, got {action_chunk.shape[1]}"
            )
        if action_chunk.shape[0] < min_horizon:
            raise PolicyServerError(
                f"action chunk horizon {action_chunk.shape[0]} < requested {min_horizon}"
            )
        if not np.all(np.isfinite(action_chunk)):
            raise PolicyServerError("action chunk contains NaN or Inf")
        return action_chunk.astype(np.float32, copy=False)

    def _flatten_action_dict(self, action: dict[str, Any]) -> np.ndarray:
        if not isinstance(action, dict):
            raise PolicyServerError(f"policy action must be a dict, got {type(action)!r}")
        configs = self.get_modality_config()
        keys = list(configs["action"].modality_keys)
        missing = [key for key in keys if key not in action]
        if missing:
            raise PolicyServerError(f"policy action missing keys: {missing}")
        chunks = []
        horizon = None
        for key in keys:
            arr = np.asarray(action[key], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim != 2:
                raise PolicyServerError(f"action[{key}] has unsupported shape {arr.shape}")
            if horizon is None:
                horizon = arr.shape[0]
            elif arr.shape[0] != horizon:
                raise PolicyServerError(
                    f"action key {key} horizon {arr.shape[0]} does not match {horizon}"
                )
            chunks.append(np.atleast_2d(arr))
        return np.concatenate(chunks, axis=1)
