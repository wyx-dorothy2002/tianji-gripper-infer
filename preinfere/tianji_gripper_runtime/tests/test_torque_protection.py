from __future__ import annotations

import unittest

import numpy as np
import pytest
from tianji_gripper_runtime.runtime.gripper_interface import GripperConnectionConfig
from tianji_gripper_runtime.runtime.gripper_interface import Rs05MitEndChannelGripperInterface
from tianji_gripper_runtime.runtime.gripper_interface import _uint_to_float
from tianji_gripper_runtime.runtime.torque_protection import TorqueGraspDetector


class _FakeEndChannel:
    def __init__(self) -> None:
        self.frames: list[str] = []

    def send_end_channel_data(self, *args: object) -> None:
        self.frames.append(str(args[1]))


class _FeedbackGripper(Rs05MitEndChannelGripperInterface):
    def __init__(
        self,
        config: GripperConnectionConfig,
        end_channel: _FakeEndChannel,
        feedback_positions: list[float],
        feedback_torque_nm: float,
    ) -> None:
        super().__init__(config, end_channel)
        self._feedback_positions = iter(feedback_positions)
        self._feedback_torque_nm = feedback_torque_nm

    def _read_latest_feedback(self) -> float | None:
        position = next(self._feedback_positions, None)
        if position is not None:
            self.last_feedback_position_rad = position
            self.last_feedback_torque_nm = self._feedback_torque_nm
            self._feedback_sequence += 1
        return position


def _frame_gains(frame_hex: str) -> tuple[float, float]:
    payload = bytes.fromhex(frame_hex)[4:]
    kp_int = ((payload[3] & 0x0F) << 8) | payload[4]
    kd_int = (payload[5] << 4) | (payload[6] >> 4)
    return (
        _uint_to_float(kp_int, 0.0, 500.0, 12),
        _uint_to_float(kd_int, 0.0, 5.0, 12),
    )


class TorqueProtectionTest(unittest.TestCase):
    def test_detector_latches_and_releases_after_torque_drops(self) -> None:
        detector = TorqueGraspDetector(
            enabled=True,
            alpha=1.0,
            torque_threshold_nm=1.0,
            release_threshold_nm=0.2,
            count_threshold=2,
        )

        assert not detector.update(1.2, closing=True, opening=False)
        assert detector.update(1.2, closing=True, opening=False)
        assert detector.has_object
        assert not detector.update(0.1, closing=False, opening=False)
        assert not detector.has_object

    def test_rs05_latches_target_and_uses_holding_gains(self) -> None:
        end_channel = _FakeEndChannel()
        gripper = _FeedbackGripper(
            GripperConnectionConfig(
                "right",
                rs05_kp=45.0,
                rs05_kd=1.0,
                rs05_min_pos_rad=-5.0,
                rs05_max_pos_rad=5.0,
                rs05_torque_protection_enabled=True,
                rs05_torque_filter_alpha=1.0,
                rs05_torque_threshold_nm=1.0,
                rs05_torque_count_threshold=2,
                rs05_holding_kp=4.0,
                rs05_holding_kd=0.1,
            ),
            end_channel,
            feedback_positions=[0.0, 0.0],
            feedback_torque_nm=1.2,
        )

        gripper.send_position(np.asarray([1.0], dtype=np.float32))
        gripper.send_position(np.asarray([1.0], dtype=np.float32))

        assert gripper.torque_protection_has_object
        assert gripper.get_position()[0] == pytest.approx(0.0)
        holding_kp, holding_kd = _frame_gains(end_channel.frames[-1])
        assert holding_kp == pytest.approx(4.0, abs=0.2)
        assert holding_kd == pytest.approx(0.1, abs=0.01)

        gripper.send_position(np.asarray([-1.0], dtype=np.float32))

        assert not gripper.torque_protection_has_object
        normal_kp, normal_kd = _frame_gains(end_channel.frames[-1])
        assert normal_kp == pytest.approx(45.0, abs=0.2)
        assert normal_kd == pytest.approx(1.0, abs=0.01)

    def test_rs05_uses_feedback_consumed_by_state_read(self) -> None:
        end_channel = _FakeEndChannel()
        gripper = _FeedbackGripper(
            GripperConnectionConfig(
                "right",
                rs05_min_pos_rad=-5.0,
                rs05_max_pos_rad=5.0,
                rs05_torque_protection_enabled=True,
                rs05_torque_filter_alpha=1.0,
                rs05_torque_threshold_nm=1.0,
                rs05_torque_count_threshold=1,
            ),
            end_channel,
            feedback_positions=[0.0],
            feedback_torque_nm=1.2,
        )

        gripper.get_position()
        gripper.send_position(np.asarray([1.0], dtype=np.float32))

        assert gripper.torque_protection_has_object
        assert gripper.get_position()[0] == pytest.approx(0.0)
