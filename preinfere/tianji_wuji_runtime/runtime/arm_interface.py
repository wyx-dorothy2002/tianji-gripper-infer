"""Arm SDK boundary for Tianji Wuji runtime.

Hardware owners should replace FakeArmInterface or subclass ArmInterface with the
real SDK implementation. All methods use 7 joint positions in schema order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import schema

if TYPE_CHECKING:
    from .tianji_arm_system import TianjiDualArmSystem


class ArmError(RuntimeError):
    """Raised by arm SDK adapters."""


@dataclass
class ArmConnectionConfig:
    side: str
    ip: str | None = None
    port: int | None = None
    sdk_root: str | None = None
    config_path: str | None = None
    unit: str = schema.STATE_UNIT


class ArmInterface:
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass

    def shared_resource_id(self) -> object:
        return id(self)

    def get_joint_state(self) -> np.ndarray:
        """Return 7-dim joint positions."""
        raise NotImplementedError

    def send_joint_position(self, q: np.ndarray) -> None:
        raise NotImplementedError

    def stage_joint_position(self, q: np.ndarray) -> None:
        self.send_joint_position(q)

    def flush_staged_commands(self) -> None:
        return None

    def hold_position(self) -> None:
        raise NotImplementedError

    def go_home(self) -> None:
        raise NotImplementedError


class FakeArmInterface(ArmInterface):
    """In-memory arm used for dry-run and integration tests."""

    def __init__(self, config: ArmConnectionConfig) -> None:
        self.config = config
        self.connected = False
        self.q = np.zeros(schema.LEFT_ARM_DOF, dtype=np.float32)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_state(self) -> np.ndarray:
        return self.q.astype(np.float32, copy=True)

    def send_joint_position(self, q: np.ndarray) -> None:
        arr = np.asarray(q, dtype=np.float32)
        if arr.shape != (schema.LEFT_ARM_DOF,):
            raise ArmError(f"{self.config.side} arm command must have shape (7,), got {arr.shape}")
        self.q = arr.copy()

    def hold_position(self) -> None:
        return None

    def go_home(self) -> None:
        self.q[:] = 0.0


class TianjiArmInterface(ArmInterface):
    """Side-specific view over a shared Tianji dual-arm host controller."""

    def __init__(self, config: ArmConnectionConfig, system: TianjiDualArmSystem) -> None:
        self.config = config
        self.system = system

    def connect(self) -> None:
        self.system.connect()

    def disconnect(self) -> None:
        self.system.disconnect()

    def shared_resource_id(self) -> object:
        return self.system.shared_resource_id()

    def get_joint_state(self) -> np.ndarray:
        return self.system.get_joint_state(self.config.side)

    def send_joint_position(self, q: np.ndarray) -> None:
        self.stage_joint_position(q)
        self.flush_staged_commands()

    def stage_joint_position(self, q: np.ndarray) -> None:
        self.system.stage_joint_position(self.config.side, q)

    def flush_staged_commands(self) -> None:
        self.system.flush_staged_commands()

    def hold_position(self) -> None:
        self.system.hold_position()

    def go_home(self) -> None:
        self.system.go_home()
