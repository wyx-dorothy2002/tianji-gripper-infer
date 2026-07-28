from __future__ import annotations

import unittest

import numpy as np

from tianji_gripper_runtime.runtime.action_adapter import (
    ActionAdapter,
    ActionAdapterError,
    RightArmGripperAction,
)
from tianji_gripper_runtime.runtime.gripper_interface import (
    FakeGripperInterface,
    GripperConnectionConfig,
)
from tianji_gripper_runtime.runtime.observation_builder import (
    ObservationBuilder,
    PiObservationBuilder,
)
from tianji_gripper_runtime.runtime.robot_interface import (
    RightArmGripperRobot,
    RobotError,
)
from tianji_gripper_runtime.runtime.robot_state import (
    RightArmGripperState,
    RobotStateError,
)
from tianji_gripper_runtime.runtime.safety import SafetyConfig, SafetyLayer
from tianji_wuji_runtime.runtime.arm_interface import ArmConnectionConfig, FakeArmInterface


class _Modality:
    def __init__(self, keys: list[str]) -> None:
        self.modality_keys = keys
        self.delta_indices = [0]


def _make_fake_dual_robot() -> tuple[
    RightArmGripperRobot,
    FakeArmInterface,
    FakeArmInterface,
    FakeGripperInterface,
    FakeGripperInterface,
]:
    left_arm = FakeArmInterface(ArmConnectionConfig("left"))
    right_arm = FakeArmInterface(ArmConnectionConfig("right"))
    left_gripper = FakeGripperInterface(GripperConnectionConfig("left"))
    right_gripper = FakeGripperInterface(GripperConnectionConfig("right"))
    robot = RightArmGripperRobot(
        left_arm,
        right_arm,
        left_gripper,
        right_gripper,
        command_left_side=True,
    )
    return robot, left_arm, right_arm, left_gripper, right_gripper


class DualArmSymmetryTest(unittest.TestCase):
    def test_dual_mode_reads_live_left_feedback_and_commands_both_sides(self) -> None:
        robot, left_arm, right_arm, left_gripper, right_gripper = _make_fake_dual_robot()
        left_arm.q = np.asarray([121, -89, -91, -119, 1, 2, 3], dtype=np.float32)
        right_arm.q = np.asarray([-129, -90, 93, -123, -1, -2, -3], dtype=np.float32)
        left_gripper.q = np.asarray([-4.0], dtype=np.float32)
        right_gripper.q = np.asarray([-5.0], dtype=np.float32)
        robot.connect()

        first = robot.get_state()
        np.testing.assert_array_equal(first.left_arm_q, left_arm.q)
        np.testing.assert_array_equal(first.right_arm_q, right_arm.q)
        np.testing.assert_array_equal(first.left_gripper_q, left_gripper.q)
        np.testing.assert_array_equal(first.right_gripper_q, right_gripper.q)
        self.assertEqual(robot.freeze_targets(), {"left_arm": None, "left_gripper": None})

        left_arm.q = left_arm.q + 0.5
        left_gripper.q = left_gripper.q + 0.25
        second = robot.get_state()
        np.testing.assert_array_equal(second.left_arm_q, left_arm.q)
        np.testing.assert_array_equal(second.left_gripper_q, left_gripper.q)

        action = RightArmGripperAction(
            right_arm_q=np.full(7, -10.0, dtype=np.float32),
            left_arm_q=np.full(7, 10.0, dtype=np.float32),
            right_gripper_q=np.asarray([-3.0], dtype=np.float32),
            left_gripper_q=np.asarray([-2.0], dtype=np.float32),
            control_left_arm=True,
            control_left_gripper=True,
        )
        robot.send_action(action)
        np.testing.assert_array_equal(left_arm.q, action.left_arm_q)
        np.testing.assert_array_equal(right_arm.q, action.right_arm_q)
        np.testing.assert_array_equal(left_gripper.q, action.left_gripper_q)
        np.testing.assert_array_equal(right_gripper.q, action.right_gripper_q)

    def test_dual_mode_rejects_freeze_targets_and_incomplete_actions(self) -> None:
        _, left_arm, right_arm, left_gripper, right_gripper = _make_fake_dual_robot()
        with self.assertRaises(RobotError):
            RightArmGripperRobot(
                left_arm,
                right_arm,
                left_gripper,
                right_gripper,
                command_left_side=True,
                left_arm_freeze_q=np.zeros(7, dtype=np.float32),
            )

        robot, *_ = _make_fake_dual_robot()
        robot.connect()
        with self.assertRaises(RobotError):
            robot.send_action(
                RightArmGripperAction(
                    right_arm_q=np.zeros(7, dtype=np.float32),
                    right_gripper_q=np.zeros(1, dtype=np.float32),
                )
            )

    def test_missing_left_values_cannot_silently_become_zero_in_dual_mode(self) -> None:
        with self.assertRaises(RobotStateError):
            RightArmGripperState(
                right_arm_q=np.zeros(7, dtype=np.float32),
                right_gripper_q=np.zeros(1, dtype=np.float32),
                include_left=True,
            )
        with self.assertRaises(ActionAdapterError):
            RightArmGripperAction(
                right_arm_q=np.zeros(7, dtype=np.float32),
                right_gripper_q=np.zeros(1, dtype=np.float32),
                control_left_arm=True,
                control_left_gripper=True,
            )

    def test_pi_and_generic_observations_convert_both_arms_equally(self) -> None:
        state_deg = np.asarray(
            [-140, -90, 90, -120, 1, 2, 3, 130, -89, -90, -121, -1, 4, 2, -5, -4],
            dtype=np.float32,
        )
        pi_builder = PiObservationBuilder(
            robot_arm_state_unit="deg",
            policy_arm_state_unit="rad",
        )
        converted = pi_builder._convert_state(state_deg)
        np.testing.assert_allclose(converted[:14], np.deg2rad(state_deg[:14]), atol=1e-6)

        generic_builder = ObservationBuilder(
            {
                "video": _Modality(["head", "left_wrist", "right_wrist"]),
                "state": _Modality(["right_arm", "left_arm"]),
                "language": _Modality(["prompt"]),
            },
            robot_arm_state_unit="deg",
            policy_arm_state_unit="rad",
        )
        structured = RightArmGripperState.from_flat(state_deg)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        observation = generic_builder.build(
            structured,
            {"head": image, "left_wrist": image, "right_wrist": image},
            "test",
        )
        np.testing.assert_allclose(
            observation["state"]["right_arm"][0, 0],
            np.deg2rad(state_deg[:7]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            observation["state"]["left_arm"][0, 0],
            np.deg2rad(state_deg[7:14]),
            atol=1e-6,
        )

    def test_action_layout_and_safety_treat_both_arms_equally(self) -> None:
        adapter = ActionAdapter(
            policy_arm_unit="rad",
            control_arm_unit="deg",
            policy_gripper_unit="rad",
            control_gripper_unit="rad",
            control_mode="dual_arm_dual_gripper",
        )
        action_deg = np.asarray(
            [-10, -20, -30, -40, -50, -60, -70, 10, 20, 30, 40, 50, 60, 70],
            dtype=np.float32,
        )
        policy_action = np.concatenate(
            [np.deg2rad(action_deg), np.asarray([-1.0, -2.0], dtype=np.float32)]
        )
        action = adapter.split_action(policy_action)
        np.testing.assert_allclose(action.right_arm_q, action_deg[:7], atol=1e-5)
        np.testing.assert_allclose(action.left_arm_q, action_deg[7:14], atol=1e-5)
        np.testing.assert_array_equal(action.right_gripper_q, [-1.0])
        np.testing.assert_array_equal(action.left_gripper_q, [-2.0])

        limits = SafetyConfig.from_yaml(
            "/home/user/workspace/TJ-gripper_infer-main/"
            "preinfere/tianji_gripper_runtime/configs/robot_limits.yaml"
        )
        limits.arm_max_step = 1.0
        limits.enable_arm_velocity_limit = False
        safety = SafetyLayer(limits, adapter)
        current = RightArmGripperState(
            right_arm_q=np.zeros(7, dtype=np.float32),
            left_arm_q=np.zeros(7, dtype=np.float32),
            right_gripper_q=np.asarray([-1.0], dtype=np.float32),
            left_gripper_q=np.asarray([-2.0], dtype=np.float32),
            include_left=True,
        )
        symmetric_target = RightArmGripperAction(
            right_arm_q=np.full(7, 5.0, dtype=np.float32),
            left_arm_q=np.full(7, 5.0, dtype=np.float32),
            right_gripper_q=np.asarray([-1.0], dtype=np.float32),
            left_gripper_q=np.asarray([-2.0], dtype=np.float32),
            control_left_arm=True,
            control_left_gripper=True,
        )
        processed, events = safety.process_chunk(current, [symmetric_target], 0.05)
        np.testing.assert_array_equal(processed[0].right_arm_q, np.ones(7, dtype=np.float32))
        np.testing.assert_array_equal(processed[0].left_arm_q, np.ones(7, dtype=np.float32))
        self.assertEqual(
            {event["segment"] for event in events if event["type"] == "delta_clip"},
            {"right_arm", "left_arm"},
        )


if __name__ == "__main__":
    unittest.main()
