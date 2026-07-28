from __future__ import annotations

import numpy as np
import pytest
from tianji_gripper_runtime.runtime.action_adapter import ActionAdapter
from tianji_gripper_runtime.runtime.gripper_normalization import GripperCalibration
from tianji_gripper_runtime.runtime.observation_builder import PiObservationBuilder


def _calibrations() -> tuple[GripperCalibration, GripperCalibration]:
    right = GripperCalibration(close_motor_rad=0.0, open_motor_rad=-2.0, label="right")
    left = GripperCalibration(close_motor_rad=1.0, open_motor_rad=3.0, label="left")
    return right, left


def test_normalization_matches_dataset_convention() -> None:
    right, left = _calibrations()

    assert right.motor_to_normalized(-2.0) == pytest.approx(0.0)
    assert right.motor_to_normalized(-1.0) == pytest.approx(0.5)
    assert right.motor_to_normalized(0.0) == pytest.approx(1.0)
    assert left.normalized_to_motor(0.0) == pytest.approx(3.0)
    assert left.normalized_to_motor(1.0) == pytest.approx(1.0)


def test_action_and_observation_use_separate_gripper_calibrations() -> None:
    right, left = _calibrations()
    adapter = ActionAdapter(
        policy_gripper_unit="normalized",
        control_gripper_unit="rad",
        control_mode="dual_arm_dual_gripper",
        right_gripper_calibration=right,
        left_gripper_calibration=left,
    )
    action = adapter.split_action(
        np.concatenate(
            [
                np.zeros(14, dtype=np.float32),
                np.asarray([0.25, 0.75], dtype=np.float32),
            ]
        )
    )
    np.testing.assert_allclose(action.right_gripper_q, [-1.5])
    np.testing.assert_allclose(action.left_gripper_q, [1.5])

    observation_builder = PiObservationBuilder(
        robot_gripper_state_unit="rad",
        policy_gripper_state_unit="normalized",
        right_gripper_calibration=right,
        left_gripper_calibration=left,
    )
    converted = observation_builder.build(
        state_pi=np.concatenate(
            [
                np.zeros(14, dtype=np.float32),
                np.asarray([-1.0, 2.0], dtype=np.float32),
            ]
        ),
        images={
            "head": np.zeros((2, 2, 3), dtype=np.uint8),
            "left_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            "right_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        task="test",
    )["state"]
    np.testing.assert_allclose(converted[14:], [0.5, 0.5])


def test_normalized_conversion_requires_calibration() -> None:
    adapter = ActionAdapter(
        policy_gripper_unit="normalized",
        control_gripper_unit="rad",
        control_mode="right_arm_right_gripper",
    )

    with pytest.raises(ValueError, match="close_motor_rad and open_motor_rad"):
        adapter.split_action(np.zeros(8, dtype=np.float32))
