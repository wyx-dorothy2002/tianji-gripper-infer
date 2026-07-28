from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "deployment" / "replay_recorded_actions.py"
)
SPEC = importlib.util.spec_from_file_location("replay_recorded_actions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def test_gripper_normalization_uses_episode_calibration() -> None:
    calibration = {"close_motor_rad": -0.1, "open_motor_rad": -5.1}
    actual = replay._gripper_motor_radians(
        np.asarray([0.0, 0.25, 1.0], dtype=np.float32), calibration
    )
    np.testing.assert_allclose(actual, [-5.1, -3.85, -0.1], atol=1e-6)


def test_dispatch_action_maps_right_then_left_and_converts_only_arms() -> None:
    class Robot:
        sent = None

        def send_action(self, action) -> None:
            self.sent = action

    robot = Robot()
    command = np.arange(16, dtype=np.float32) / 10.0
    replay.dispatch_action(command, robot)

    np.testing.assert_allclose(robot.sent.right_arm_q, np.rad2deg(command[:7]))
    np.testing.assert_allclose(robot.sent.left_arm_q, np.rad2deg(command[7:14]))
    np.testing.assert_allclose(robot.sent.right_gripper_q, command[14:15])
    np.testing.assert_allclose(robot.sent.left_gripper_q, command[15:16])
    assert robot.sent.control_left_arm
    assert robot.sent.control_left_gripper


def test_runtime_trace_layout_is_mapped_to_policy_layout(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    runtime = episode / "_runtime"
    runtime.mkdir(parents=True)
    (episode / "meta.json").write_text("{}", encoding="utf-8")
    header = replay.TRACE_COLUMNS
    row = [100.0, *range(1, 8), *range(11, 18), -4.2, -4.3]
    (runtime / "marvin_drag_teach_teleop_trace.csv").write_text(
        ",".join(header) + "\n" + ",".join(map(str, row)) + "\n",
        encoding="utf-8",
    )

    actions, times, _ = replay.load_trajectory(episode)

    np.testing.assert_allclose(actions[0, :7], np.deg2rad(range(1, 8)))
    np.testing.assert_allclose(actions[0, 7:14], np.deg2rad(range(11, 18)))
    np.testing.assert_allclose(actions[0, 14:], [-4.2, -4.3])
    np.testing.assert_allclose(times, [0.0])
