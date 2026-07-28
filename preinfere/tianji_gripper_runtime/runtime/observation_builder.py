"""Build GR00T Policy API observations for Tianji gripper checkpoints."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import schema
from .gripper_normalization import GripperCalibration
from .gripper_normalization import convert_gripper_units
from .robot_state import RightArmGripperState, ensure_state


class ObservationError(RuntimeError):
    """Raised when observations cannot be built safely."""


def validate_policy_inputs(
    state: RightArmGripperState | np.ndarray,
    images: Mapping[str, np.ndarray],
    *,
    required_camera_keys: tuple[str, ...] | list[str],
) -> RightArmGripperState:
    try:
        checked_state = ensure_state(state)
    except Exception as exc:  # noqa: BLE001
        raise ObservationError(f"invalid robot state before policy inference: {exc}") from exc
    for name, values, dim in (
        ("right_arm", checked_state.right_arm_q, schema.RIGHT_ARM_DOF),
        ("left_arm", checked_state.left_arm_q, schema.LEFT_ARM_DOF),
        ("right_gripper", checked_state.right_gripper_q, schema.RIGHT_GRIPPER_DOF),
        ("left_gripper", checked_state.left_gripper_q, schema.LEFT_GRIPPER_DOF),
    ):
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


class ObservationBuilder:
    def __init__(
        self,
        modality_configs: Mapping[str, Any],
        *,
        state_key_map: Mapping[str, str] | None = None,
        camera_key_map: Mapping[str, str] | None = None,
        robot_arm_state_unit: str = schema.STATE_UNIT,
        policy_arm_state_unit: str = schema.STATE_UNIT,
        robot_gripper_state_unit: str = "rad",
        policy_gripper_state_unit: str = "rad",
        right_gripper_calibration: GripperCalibration | None = None,
        left_gripper_calibration: GripperCalibration | None = None,
    ) -> None:
        self.modality_configs = modality_configs
        self.state_key_map = dict(state_key_map or {})
        self.camera_key_map = dict(camera_key_map or {})
        self.robot_arm_state_unit = robot_arm_state_unit
        self.policy_arm_state_unit = policy_arm_state_unit
        self.robot_gripper_state_unit = robot_gripper_state_unit
        self.policy_gripper_state_unit = policy_gripper_state_unit
        self.right_gripper_calibration = right_gripper_calibration
        self.left_gripper_calibration = left_gripper_calibration
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
        state: RightArmGripperState | np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> dict[str, Any]:
        checked_state = ensure_state(state)
        right_arm_q = _convert_units(
            checked_state.right_arm_q,
            self.robot_arm_state_unit,
            self.policy_arm_state_unit,
        )
        left_arm_q = _convert_units(
            checked_state.left_arm_q,
            self.robot_arm_state_unit,
            self.policy_arm_state_unit,
        )
        right_gripper_q = convert_gripper_units(
            checked_state.right_gripper_q,
            self.robot_gripper_state_unit,
            self.policy_gripper_state_unit,
            self.right_gripper_calibration,
        )
        left_gripper_q = (
            convert_gripper_units(
                checked_state.left_gripper_q,
                self.robot_gripper_state_unit,
                self.policy_gripper_state_unit,
                self.left_gripper_calibration,
            )
            if checked_state.include_left
            else checked_state.left_gripper_q.copy()
        )
        state_vector = RightArmGripperState(
            right_arm_q=right_arm_q,
            left_arm_q=left_arm_q,
            right_gripper_q=right_gripper_q,
            left_gripper_q=left_gripper_q,
            include_left=checked_state.include_left,
        ).as_flat(control_mode="dual_arm_dual_gripper" if checked_state.include_left else "right_arm_right_gripper")
        if not task or not task.strip():
            raise ObservationError("task must be a non-empty string")

        for model_key, source_key in zip(self.video_keys, self.required_camera_keys):
            if source_key not in images:
                raise ObservationError(f"missing required camera image {source_key!r}")
            self._image_buffers[model_key].append(_validate_rgb_image(images[source_key], source_key))
        self._state_buffer.append(state_vector)

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
        state_keys = (
            schema.FULL_STATE_KEYS
            if state_history.shape[1] == schema.FULL_STATE_DIM
            else schema.RIGHT_ONLY_STATE_KEYS
        )
        if mapped in state_keys:
            idx = state_keys.index(mapped)
            return state_history[:, idx : idx + 1]
            raise ObservationError(
                f"do not know how to map model state key {key!r} to the Tianji state schema"
            )

    @staticmethod
    def _stack_history(buffer: deque[np.ndarray], horizon: int, name: str) -> np.ndarray:
        if not buffer:
            raise ObservationError(f"{name} history buffer is empty")
        values = list(buffer)
        while len(values) < horizon:
            values.insert(0, values[0])
        return np.stack(values[-horizon:], axis=0)


def _convert_units(arr: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    from_unit = str(from_unit).lower()
    to_unit = str(to_unit).lower()
    values = np.asarray(arr, dtype=np.float32)
    if from_unit == to_unit:
        return values.astype(np.float32, copy=True)
    if from_unit == "rad" and to_unit in {"deg", "degree", "degrees"}:
        return np.rad2deg(values).astype(np.float32)
    if from_unit in {"deg", "degree", "degrees"} and to_unit == "rad":
        return np.deg2rad(values).astype(np.float32)
    return values.astype(np.float32, copy=True)


def _validate_rgb_image(image: np.ndarray, key: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ObservationError(f"image {key} must be uint8 RGB, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ObservationError(f"image {key} must be HxWx3 RGB, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ObservationError(f"image {key} contains non-finite values")
    return np.ascontiguousarray(arr)


class PiObservationBuilder:
    """Build openpi/pi0.5 observations for the ziyi 15D checkpoint."""

    def __init__(
        self,
        *,
        robot_arm_state_unit: str = schema.STATE_UNIT,
        policy_arm_state_unit: str = "rad",
        robot_gripper_state_unit: str = "rad",
        policy_gripper_state_unit: str = "rad",
        right_gripper_calibration: GripperCalibration | None = None,
        left_gripper_calibration: GripperCalibration | None = None,
    ) -> None:
        self.robot_arm_state_unit = robot_arm_state_unit
        self.policy_arm_state_unit = policy_arm_state_unit
        self.robot_gripper_state_unit = robot_gripper_state_unit
        self.policy_gripper_state_unit = policy_gripper_state_unit
        self.right_gripper_calibration = right_gripper_calibration
        self.left_gripper_calibration = left_gripper_calibration
        self.video_keys = ["head", "left_wrist", "right_wrist"]
        self.required_camera_keys = list(self.video_keys)

    def build(
        self,
        *,
        state_pi: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> dict[str, Any]:
        state = self._convert_state(state_pi)
        if not task or not task.strip():
            raise ObservationError("task must be a non-empty string")
        missing = [key for key in self.required_camera_keys if key not in images]
        if missing:
            raise ObservationError(f"missing required pi camera image(s): {missing}")
        return {
            "state": state,
            "images": {
                "head": _validate_rgb_image(images["head"], "head"),
                "left_wrist": _validate_rgb_image(images["left_wrist"], "left_wrist"),
                "right_wrist": _validate_rgb_image(images["right_wrist"], "right_wrist"),
            },
            "prompt": task.strip(),
        }

    def _convert_state(self, state_pi: np.ndarray) -> np.ndarray:
        arr = np.asarray(state_pi, dtype=np.float32).reshape(-1)
        if arr.shape not in {(15,), (16,)}:
            raise ObservationError(f"pi state must have shape (15,) or (16,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ObservationError("pi state contains NaN or Inf")
        converted = arr.copy()
        converted[0:7] = _convert_units(
            converted[0:7],
            self.robot_arm_state_unit,
            self.policy_arm_state_unit,
        )
        converted[7:14] = _convert_units(
            converted[7:14],
            self.robot_arm_state_unit,
            self.policy_arm_state_unit,
        )
        converted[14:15] = convert_gripper_units(
            converted[14:15],
            self.robot_gripper_state_unit,
            self.policy_gripper_state_unit,
            self.right_gripper_calibration,
        )
        if converted.shape == (16,):
            converted[15:16] = convert_gripper_units(
                converted[15:16],
                self.robot_gripper_state_unit,
                self.policy_gripper_state_unit,
                self.left_gripper_calibration,
            )
        return converted.astype(np.float32, copy=False)
