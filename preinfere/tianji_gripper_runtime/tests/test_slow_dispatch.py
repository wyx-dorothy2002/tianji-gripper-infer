from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pytest
from tianji_gripper_runtime.runtime.action_adapter import ActionAdapter
from tianji_gripper_runtime.runtime.action_adapter import RightArmGripperAction
from tianji_gripper_runtime.runtime.executor import ActionExecutor
from tianji_gripper_runtime.runtime.gripper_interface import FakeGripperInterface
from tianji_gripper_runtime.runtime.gripper_interface import GripperConnectionConfig
from tianji_gripper_runtime.runtime.robot_interface import RightArmGripperRobot
from tianji_gripper_runtime.runtime.slow_dispatch import SlowDispatchCancelledError
from tianji_gripper_runtime.runtime.slow_dispatch import SlowDispatchConfig
from tianji_gripper_runtime.runtime.slow_dispatch import SlowInterpolatedDispatcher
from tianji_wuji_runtime.runtime.arm_interface import ArmConnectionConfig
from tianji_wuji_runtime.runtime.arm_interface import FakeArmInterface


def _make_fake_robot() -> RightArmGripperRobot:
    return RightArmGripperRobot(
        FakeArmInterface(ArmConnectionConfig("left")),
        FakeArmInterface(ArmConnectionConfig("right")),
        FakeGripperInterface(GripperConnectionConfig("left")),
        FakeGripperInterface(GripperConnectionConfig("right")),
        command_left_side=True,
    )


class SlowDispatchTest(unittest.TestCase):
    def test_loads_timing_from_inference_yaml(self) -> None:
        config = SlowDispatchConfig.from_yaml(
            Path(__file__).resolve().parents[1] / "configs" / "infer.yaml"
        )

        assert config.duration_sec == 1.0
        assert config.frequency_hz == 20.0
        assert config.interpolation_mode == "cubic"

    def test_interpolates_both_arms_and_both_grippers(self) -> None:
        robot = _make_fake_robot()
        robot.connect()
        sent: list[object] = []
        original_send_action = robot.send_action

        def capture(action) -> None:
            sent.append(action.copy())
            original_send_action(action)

        robot.send_action = capture
        adapter = ActionAdapter(
            policy_arm_unit="rad",
            control_arm_unit="deg",
            policy_gripper_unit="rad",
            control_gripper_unit="rad",
            control_mode="dual_arm_dual_gripper",
        )
        dispatcher = SlowInterpolatedDispatcher(
            robot,
            adapter=adapter,
            duration_sec=0.1,
            frequency_hz=20.0,
            interpolation_mode="linear",
        )
        model_action = np.concatenate(
            [
                np.full(7, np.deg2rad(10.0), dtype=np.float32),
                np.full(7, np.deg2rad(-10.0), dtype=np.float32),
                np.asarray([1.0, -1.0], dtype=np.float32),
            ]
        )

        target = dispatcher.dispatch(model_action)

        assert len(sent) == 2
        np.testing.assert_allclose(sent[0].right_arm_q, 5.0)
        np.testing.assert_allclose(sent[0].left_arm_q, -5.0)
        np.testing.assert_allclose(sent[0].right_gripper_q, 0.5)
        np.testing.assert_allclose(sent[0].left_gripper_q, -0.5)
        np.testing.assert_allclose(sent[-1].right_arm_q, 10.0)
        np.testing.assert_allclose(sent[-1].left_arm_q, -10.0)
        np.testing.assert_allclose(sent[-1].right_gripper_q, 1.0)
        np.testing.assert_allclose(sent[-1].left_gripper_q, -1.0)
        np.testing.assert_allclose(target.right_arm_q, 10.0)
        np.testing.assert_allclose(target.left_arm_q, -10.0)

    def test_stop_holds_and_cancels_before_sending(self) -> None:
        robot = _make_fake_robot()
        robot.connect()
        adapter = ActionAdapter(control_mode="dual_arm_dual_gripper")
        dispatcher = SlowInterpolatedDispatcher(robot, adapter=adapter, duration_sec=1.0)

        with pytest.raises(SlowDispatchCancelledError):
            dispatcher.dispatch(np.ones(16, dtype=np.float32), stop_callback=lambda: True)

        state = robot.get_state()
        np.testing.assert_array_equal(state.right_arm_q, np.zeros(7, dtype=np.float32))
        np.testing.assert_array_equal(state.left_arm_q, np.zeros(7, dtype=np.float32))
        np.testing.assert_array_equal(state.right_gripper_q, np.zeros(1, dtype=np.float32))
        np.testing.assert_array_equal(state.left_gripper_q, np.zeros(1, dtype=np.float32))

    def test_dispatches_direct_control_target_for_reset(self) -> None:
        robot = _make_fake_robot()
        robot.connect()
        sent: list[object] = []
        original_send_action = robot.send_action

        def capture(action) -> None:
            sent.append(action.copy())
            original_send_action(action)

        robot.send_action = capture
        dispatcher = SlowInterpolatedDispatcher(
            robot,
            adapter=ActionAdapter(control_mode="dual_arm_dual_gripper"),
            duration_sec=0.1,
            frequency_hz=20.0,
            interpolation_mode="linear",
        )
        target = RightArmGripperAction(
            right_arm_q=np.asarray([-140, -90, 90, -120, 0, 0, 0], dtype=np.float32),
            left_arm_q=np.asarray([140, -90, -90, -120, 0, 0, 0], dtype=np.float32),
            right_gripper_q=np.asarray([-5.2817], dtype=np.float32),
            left_gripper_q=np.asarray([-5.0616], dtype=np.float32),
            control_left_arm=True,
            control_left_gripper=True,
        )

        final_target = dispatcher.dispatch_control_action(target)

        assert len(sent) == 2
        np.testing.assert_allclose(sent[-1].right_arm_q, target.right_arm_q)
        np.testing.assert_allclose(sent[-1].left_arm_q, target.left_arm_q)
        np.testing.assert_allclose(sent[-1].right_gripper_q, target.right_gripper_q)
        np.testing.assert_allclose(sent[-1].left_gripper_q, target.left_gripper_q)
        np.testing.assert_allclose(final_target.right_arm_q, target.right_arm_q)

    def test_default_executor_uses_slow_mode_only_when_enabled(self) -> None:
        robot = _make_fake_robot()
        robot.connect()
        sent: list[object] = []
        original_send_action = robot.send_action

        def capture(action) -> None:
            sent.append(action.copy())
            original_send_action(action)

        robot.send_action = capture
        action = RightArmGripperAction(
            right_arm_q=np.full(7, 10.0, dtype=np.float32),
            left_arm_q=np.full(7, -10.0, dtype=np.float32),
            right_gripper_q=np.asarray([1.0], dtype=np.float32),
            left_gripper_q=np.asarray([-1.0], dtype=np.float32),
            control_left_arm=True,
            control_left_gripper=True,
        )
        executor = ActionExecutor(
            robot,
            adapter=ActionAdapter(control_mode="dual_arm_dual_gripper"),
            slow_dispatch_enabled=True,
            slow_dispatch_duration_sec=0.1,
            slow_dispatch_frequency_hz=20.0,
            slow_dispatch_interpolation_mode="linear",
        )

        executor.execute_chunk([action], dt=0.05)

        assert len(sent) == 2
        np.testing.assert_allclose(sent[0].right_arm_q, 5.0)
        np.testing.assert_allclose(sent[0].left_arm_q, -5.0)
        np.testing.assert_allclose(sent[0].right_gripper_q, 0.5)
        np.testing.assert_allclose(sent[0].left_gripper_q, -0.5)
