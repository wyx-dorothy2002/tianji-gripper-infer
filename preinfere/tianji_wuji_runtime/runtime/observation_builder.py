"""Build GR00T Policy API observations from robot state, images, and task text."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import schema
from .robot_state import DualArmHandState, ensure_state


class ObservationError(RuntimeError):
    """Raised when observations cannot be built safely."""


# Fixed GR00T policy input contract for the Tianji Wuji checkpoint (delta_indices=[0]).
VIDEO_KEYS = ("head", "left_wrist", "right_wrist")
STATE_GROUP_KEYS = {
    "left_arm_joint": "left_arm",
    "right_arm_joint": "right_arm",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
}


def validate_policy_inputs(
    state: DualArmHandState | np.ndarray,
    images: Mapping[str, np.ndarray],
    *,
    required_camera_keys: tuple[str, ...] | list[str],
) -> DualArmHandState:
    """Validate the full robot-state + RGB-image input before policy inference."""
    try:
        checked_state = ensure_state(state)
    except Exception as exc:  # noqa: BLE001
        raise ObservationError(f"invalid robot state before policy inference: {exc}") from exc

    state_segments = {
        "left_arm": (checked_state.left_arm_q, schema.LEFT_ARM_DOF),
        "right_arm": (checked_state.right_arm_q, schema.RIGHT_ARM_DOF),
        "left_hand": (checked_state.left_hand_q, schema.LEFT_HAND_DOF),
        "right_hand": (checked_state.right_hand_q, schema.RIGHT_HAND_DOF),
    }
    for name, (values, dim) in state_segments.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (dim,):
            raise ObservationError(f"robot state {name} must have shape ({dim},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ObservationError(f"robot state {name} contains NaN or Inf")

    missing = [key for key in required_camera_keys if key not in images]
    if missing:
        raise ObservationError(f"missing camera image(s) before policy inference: {missing}")
    for key in required_camera_keys:
        _validate_rgb_image(images[key], key)
    return checked_state


def build_policy_observation(
    state: "DualArmHandState | np.ndarray",
    images: Mapping[str, np.ndarray],
    *,
    video_keys: tuple[str, ...] = VIDEO_KEYS,
) -> dict[str, np.ndarray]:
    """Build a policy-ready observation dict using the fixed contract.

    Unlike :class:`ObservationBuilder`, this needs no policy-server modality config,
    so offline tools (e.g. the observation capture script) can produce arrays whose
    shapes match the live policy input:

        video.<key>          -> uint8  (1, 1, H, W, 3)
        state.left_arm_joint -> float32 (1, 1, 7)
        state.right_arm_joint-> float32 (1, 1, 7)
        state.left_hand      -> float32 (1, 1, 20)
        state.right_hand     -> float32 (1, 1, 20)
    """
    flat = ensure_state(state).as_flat()
    out: dict[str, np.ndarray] = {}
    for key in video_keys:
        if key not in images:
            raise ObservationError(f"missing required camera image {key!r}")
        out[f"video.{key}"] = _validate_rgb_image(images[key], key)[None, None, ...]
    for model_key, group in STATE_GROUP_KEYS.items():
        segment = schema.SEGMENTS[group]
        out[f"state.{model_key}"] = (
            flat[segment.vector_slice].astype(np.float32)[None, None, ...]
        )
    return out


class ObservationBuilder:
    def __init__(
        self,
        modality_configs: Mapping[str, Any],
        *,
        state_key_map: Mapping[str, str] | None = None,
        camera_key_map: Mapping[str, str] | None = None,
    ) -> None:
        self.modality_configs = modality_configs
        self.state_key_map = dict(state_key_map or {})
        self.camera_key_map = dict(camera_key_map or {})
        self.video_keys = list(modality_configs["video"].modality_keys)
        self.state_keys = list(modality_configs["state"].modality_keys)
        self.language_key = list(modality_configs["language"].modality_keys)[0]
        self.video_horizon = len(modality_configs["video"].delta_indices)
        self.state_horizon = len(modality_configs["state"].delta_indices)
        self._image_buffers: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=self.video_horizon) for key in self.video_keys
        }
        self._state_buffer: deque[np.ndarray] = deque(maxlen=self.state_horizon)

    @property
    def required_camera_keys(self) -> list[str]:
        return [self.camera_key_map.get(key, key) for key in self.video_keys]

    def build(
        self,
        state: DualArmHandState | np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> dict[str, Any]:
        state_54 = ensure_state(state).as_flat()
        if not task or not task.strip():
            raise ObservationError("task must be a non-empty string")

        for model_key, source_key in zip(self.video_keys, self.required_camera_keys):
            if source_key not in images:
                raise ObservationError(f"missing required camera image {source_key!r}")
            image = self._validate_image(images[source_key], source_key)
            self._image_buffers[model_key].append(image)
        self._state_buffer.append(state_54)

        video = {
            key: self._stack_history(self._image_buffers[key], self.video_horizon, key)[None, ...]
            for key in self.video_keys
        }

        state_history = self._stack_history(self._state_buffer, self.state_horizon, "state")
        state_dict = {
            key: self._extract_state_key(state_history, key)[None, ...].astype(np.float32)
            for key in self.state_keys
        }

        return {
            "video": video,
            "state": state_dict,
            "language": {self.language_key: [[task.strip()]]},
        }

    def _extract_state_key(self, state_history: np.ndarray, key: str) -> np.ndarray:
        mapped = self.state_key_map.get(key, key)
        if mapped in schema.GROUP_ALIASES:
            segment = schema.SEGMENTS[schema.GROUP_ALIASES[mapped]]
            return state_history[:, segment.vector_slice]
        if mapped in schema.STATE_KEYS:
            idx = schema.STATE_KEYS.index(mapped)
            return state_history[:, idx : idx + 1]
        if set(self.state_keys) == set(schema.STATE_KEYS):
            idx = schema.STATE_KEYS.index(key)
            return state_history[:, idx : idx + 1]
        if len(self.state_keys) == 1:
            return state_history
        raise ObservationError(
            f"do not know how to map model state key {key!r} to the 54-DoF schema; "
            "add it to state_key_map or align checkpoint modality keys"
        )

    @staticmethod
    def _validate_image(image: np.ndarray, key: str) -> np.ndarray:
        return _validate_rgb_image(image, key)

    @staticmethod
    def _stack_history(buffer: deque[np.ndarray], horizon: int, name: str) -> np.ndarray:
        if not buffer:
            raise ObservationError(f"{name} history buffer is empty")
        values = list(buffer)
        while len(values) < horizon:
            values.insert(0, values[0])
        return np.stack(values[-horizon:], axis=0)


def _validate_rgb_image(image: np.ndarray, key: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ObservationError(f"image {key} must be uint8 RGB, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ObservationError(f"image {key} must be HxWx3 RGB, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ObservationError(f"image {key} contains non-finite values")
    return np.ascontiguousarray(arr)
