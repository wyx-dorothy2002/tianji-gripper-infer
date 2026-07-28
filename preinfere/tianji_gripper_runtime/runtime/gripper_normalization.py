"""Shared gripper motor-angle normalization used by policy I/O."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class GripperNormalizationError(ValueError):
    """Raised when gripper normalization calibration is missing or invalid."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


@dataclass(frozen=True)
class GripperCalibration:
    """Motor-angle endpoints with the dataset convention closed=1, open=0."""

    close_motor_rad: float | None = None
    open_motor_rad: float | None = None
    label: str = "gripper"

    def _span(self) -> float:
        if self.close_motor_rad is None or self.open_motor_rad is None:
            raise GripperNormalizationError(
                f"{self.label} normalization requires close_motor_rad and open_motor_rad"
            )
        close = float(self.close_motor_rad)
        opening = float(self.open_motor_rad)
        span = opening - close
        if not math.isfinite(close) or not math.isfinite(opening) or not math.isfinite(span):
            raise GripperNormalizationError(
                f"{self.label} normalization calibration must contain finite values"
            )
        if abs(span) <= 1e-9:
            raise GripperNormalizationError(
                f"{self.label} close_motor_rad and open_motor_rad must be distinct"
            )
        return span

    def motor_to_normalized(self, motor_rad: float) -> float:
        span = self._span()
        normalized = (float(self.open_motor_rad) - float(motor_rad)) / span
        return _clamp(normalized, 0.0, 1.0)

    def normalized_to_motor(self, normalized: float) -> float:
        span = self._span()
        value = _clamp(float(normalized), 0.0, 1.0)
        return float(self.open_motor_rad) - value * span


def convert_gripper_units(
    values: np.ndarray,
    from_unit: str,
    to_unit: str,
    calibration: GripperCalibration | None,
) -> np.ndarray:
    """Convert gripper values between motor rad and normalized closed=1/open=0."""

    source = from_unit.strip().lower()
    target = to_unit.strip().lower()
    arr = np.asarray(values, dtype=np.float32)
    if source == target:
        return arr.astype(np.float32, copy=True)
    normalized_units = {"normalized", "normalised"}
    supported_units = {"rad", *normalized_units}
    if source not in supported_units or target not in supported_units:
        raise GripperNormalizationError(
            f"unsupported gripper unit conversion {from_unit!r} -> {to_unit!r}"
        )
    if calibration is None:
        raise GripperNormalizationError(
            "normalized gripper values require close_motor_rad and open_motor_rad"
        )
    if source in normalized_units:
        converted = [calibration.normalized_to_motor(float(value)) for value in arr]
    else:
        converted = [calibration.motor_to_normalized(float(value)) for value in arr]
    return np.asarray(converted, dtype=np.float32).reshape(arr.shape)
