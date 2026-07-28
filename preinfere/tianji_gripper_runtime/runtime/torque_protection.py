"""Torque-feedback grasp protection for RS05 grippers."""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


@dataclass
class TorqueGraspDetector:
    """Filter RS05 torque feedback and latch an object-held state."""

    enabled: bool = False
    mode: str = "torque"
    alpha: float = 0.3
    torque_threshold_nm: float = 1.0
    release_threshold_nm: float = 0.2
    count_threshold: int = 5
    filtered_effort_nm: float = 0.0
    counter: int = 0
    has_object: bool = False

    def update(self, torque_nm: float | None, *, closing: bool, opening: bool) -> bool:
        active = self.enabled and self.mode == "torque"
        measurement = float(torque_nm or 0.0) if active else 0.0
        alpha = _clamp(self.alpha, 0.0, 1.0)
        self.filtered_effort_nm = (
            alpha * measurement + (1.0 - alpha) * self.filtered_effort_nm
        )

        if opening or not active:
            self.reset()
            return False

        if self.has_object:
            release_threshold = max(float(self.release_threshold_nm), 0.0)
            if not closing and abs(self.filtered_effort_nm) <= release_threshold:
                self.reset()
            return self.has_object

        if not closing:
            return False

        threshold_effort = max(abs(self.filtered_effort_nm), abs(measurement))
        if threshold_effort >= max(float(self.torque_threshold_nm), 0.0):
            self.counter += 1
        else:
            self.counter = max(0, self.counter - 1)
        if self.counter >= max(int(self.count_threshold), 1):
            self.has_object = True
        return self.has_object

    def reset(self) -> None:
        self.filtered_effort_nm = 0.0
        self.counter = 0
        self.has_object = False
