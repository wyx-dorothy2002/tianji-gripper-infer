#!/usr/bin/env python3
"""Minimal Marvin/Tianji state=3 joint-command probe.

This bypasses the GR00T robot runtime and talks to the Tianji Marvin SDK directly.
It mirrors the DexProj state=3 probe path:
  clear faults -> optional idle reset -> enter state=3 with current-joint hold
  -> joint impedance + drag_space=0 -> small joint jog.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable


DEFAULT_SDK_ROOT = Path(
    "/home/user/workspace/TJ-gripper-codex-marvin-gripper-controller"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "test" / "action_replay" / "send_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.8.166")
    parser.add_argument("--tianji-sdk-root", default=str(DEFAULT_SDK_ROOT))
    parser.add_argument("--arm", choices=["A", "B"], default="B")
    parser.add_argument(
        "--joint",
        type=int,
        default=1,
        help="1-based joint index to jog on the selected arm.",
    )
    parser.add_argument("--jog-deg", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--vel-ratio", type=int, default=100)
    parser.add_argument("--acc-ratio", type=int, default=100)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument("--hold-sec", type=float, default=0.5)
    parser.add_argument("--read-every", type=int, default=5)
    parser.add_argument("--no-jog", action="store_true")
    parser.add_argument("--skip-idle-reset", action="store_true")
    parser.add_argument("--leave-enabled", action="store_true")
    parser.add_argument("--strict-sdk-return", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.joint < 1 or args.joint > 7:
        raise ValueError("--joint must be in [1, 7]")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.hz <= 0:
        raise ValueError("--hz must be positive")

    sdk_root = Path(args.tianji_sdk_root).expanduser().resolve()
    Marvin_Robot, DCSS = _load_marvin_sdk(sdk_root)
    dcss = DCSS()
    trace_path = _make_trace_path(Path(args.output_dir))

    robot = Marvin_Robot()
    connected = robot.connect(args.robot_ip)
    print(f"[probe] connect {args.robot_ip}: {connected}")
    if connected not in (True, 1):
        raise RuntimeError(f"Marvin connect failed for ip={args.robot_ip}: {connected!r}")

    try:
        initial = _read_feedback(robot, dcss)
        _print_snapshot("initial", initial)
        _append_trace(trace_path, "initial", initial)

        _clear_faults(robot, strict=args.strict_sdk_return)
        _sleep(args.settle_sec)

        if not args.skip_idle_reset:
            _write_batch(
                robot,
                ("set_state_A_0", lambda: robot.set_state(arm="A", state=0)),
                ("set_state_B_0", lambda: robot.set_state(arm="B", state=0)),
                strict=args.strict_sdk_return,
            )
            print("[probe] sent A/B state=0")
            _sleep(args.settle_sec)
            _clear_faults(robot, strict=args.strict_sdk_return)
            _sleep(args.settle_sec)

        before_enable = _read_feedback(robot, dcss)
        current = _current_joints_by_arm(before_enable)
        _print_snapshot("before_enable", before_enable)
        _append_trace(trace_path, "before_enable", before_enable)

        _enter_state3_with_current_hold(
            robot,
            current,
            vel_ratio=args.vel_ratio,
            acc_ratio=args.acc_ratio,
            strict=args.strict_sdk_return,
        )
        print(
            "[probe] sent A/B state=3, vel/acc, and current-joint hold "
            f"(vel={args.vel_ratio}, acc={args.acc_ratio})"
        )
        _sleep(args.settle_sec)

        _configure_joint_impedance(robot, strict=args.strict_sdk_return)
        print("[probe] sent impedance_type=1, joint_kd, drag_space=0")
        _sleep(args.settle_sec)

        after_enable = _read_feedback(robot, dcss)
        _print_snapshot("after_enable", after_enable, target_by_arm=current)
        _append_trace(trace_path, "after_enable", after_enable, target_by_arm=current)

        hold_targets = _current_joints_by_arm(after_enable)
        _hold_current(
            robot, hold_targets, seconds=args.hold_sec, hz=args.hz, strict=args.strict_sdk_return
        )
        after_hold = _read_feedback(robot, dcss)
        _print_snapshot("after_hold", after_hold, target_by_arm=hold_targets)
        _append_trace(trace_path, "after_hold", after_hold, target_by_arm=hold_targets)

        if not args.no_jog:
            _jog_selected_joint(
                robot,
                dcss,
                args,
                hold_targets,
                trace_path=trace_path,
                strict=args.strict_sdk_return,
            )

        print(f"[probe] trace: {trace_path}")
        return 0
    finally:
        if not args.leave_enabled:
            try:
                _write_batch(
                    robot,
                    ("set_state_A_0", lambda: robot.set_state(arm="A", state=0)),
                    ("set_state_B_0", lambda: robot.set_state(arm="B", state=0)),
                    strict=False,
                )
                print("[probe] sent A/B state=0")
                _sleep(0.5)
                after_disable = _read_feedback(robot, dcss)
                _print_snapshot("after_disable", after_disable)
                _append_trace(trace_path, "after_disable", after_disable)
            except Exception as exc:  # noqa: BLE001
                print(f"[probe] disable/readback failed: {exc}")
        try:
            robot.release_robot()
            print("[probe] released robot")
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] release failed: {exc}")


def _load_marvin_sdk(sdk_root: Path) -> tuple[type[Any], type[Any]]:
    if not sdk_root.exists():
        raise FileNotFoundError(f"Tianji SDK root does not exist: {sdk_root}")
    root = sdk_root.expanduser().resolve()

    legacy_root = root
    marvin_parent: Path | None = None
    if (root / "src/marvin.py").exists():
        marvin_parent = root / "reference/MARVIN_SDK"
    elif root.name == "MARVIN_SDK" and (root / "SDK_PYTHON").exists():
        marvin_parent = root
    elif root.name == "SDK_PYTHON":
        marvin_parent = root.parent

    if marvin_parent is not None and (marvin_parent / "SDK_PYTHON/fx_robot.py").exists():
        parent_str = str(marvin_parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
        fx_robot = importlib.import_module("SDK_PYTHON.fx_robot")
        return fx_robot.Marvin_Robot, fx_robot.DCSS

    root_str = str(legacy_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    loaded = sys.modules.get("tianji_output")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", "") or ""
        if loaded_file and not Path(loaded_file).resolve().is_relative_to(legacy_root):
            sys.modules.pop("tianji_output", None)

    fx_robot = importlib.import_module("tianji_output._internal.fx_robot")
    structure_data = importlib.import_module("tianji_output._internal.structure_data")
    return fx_robot.Marvin_Robot, structure_data.DCSS


def _make_trace_path(output_dir: Path) -> Path:
    run_dir = output_dir.expanduser().resolve() / datetime.now().strftime(
        "probe_state3_%Y%m%d_%H%M%S_%f"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir / "probe_trace.jsonl"


def _clear_faults(robot: Any, *, strict: bool) -> None:
    _write_batch(
        robot,
        ("clear_error_A", lambda: robot.clear_error("A")),
        ("clear_error_B", lambda: robot.clear_error("B")),
        strict=strict,
    )
    print("[probe] sent clear_error A/B")


def _enter_state3_with_current_hold(
    robot: Any,
    current: dict[str, list[float]],
    *,
    vel_ratio: int,
    acc_ratio: int,
    strict: bool,
) -> None:
    _write_batch(
        robot,
        ("set_state_A_3", lambda: robot.set_state(arm="A", state=3)),
        ("set_state_B_3", lambda: robot.set_state(arm="B", state=3)),
        (
            "set_vel_acc_A",
            lambda: robot.set_vel_acc(arm="A", velRatio=int(vel_ratio), AccRatio=int(acc_ratio)),
        ),
        (
            "set_vel_acc_B",
            lambda: robot.set_vel_acc(arm="B", velRatio=int(vel_ratio), AccRatio=int(acc_ratio)),
        ),
        ("hold_A_current", lambda: robot.set_joint_cmd_pose(arm="A", joints=current["A"])),
        ("hold_B_current", lambda: robot.set_joint_cmd_pose(arm="B", joints=current["B"])),
        strict=strict,
    )


def _configure_joint_impedance(robot: Any, *, strict: bool) -> None:
    joint_k = [2, 2, 2, 1.6, 1, 1, 1]
    joint_d = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]
    _write_batch(
        robot,
        ("set_impedance_type_A", lambda: robot.set_impedance_type(arm="A", type=1)),
        ("set_impedance_type_B", lambda: robot.set_impedance_type(arm="B", type=1)),
        ("set_joint_kd_A", lambda: robot.set_joint_kd_params(arm="A", K=joint_k, D=joint_d)),
        ("set_joint_kd_B", lambda: robot.set_joint_kd_params(arm="B", K=joint_k, D=joint_d)),
        ("set_drag_space_A", lambda: robot.set_drag_space(arm="A", dgType=0)),
        ("set_drag_space_B", lambda: robot.set_drag_space(arm="B", dgType=0)),
        strict=strict,
    )


def _hold_current(
    robot: Any,
    targets: dict[str, list[float]],
    *,
    seconds: float,
    hz: float,
    strict: bool,
) -> None:
    count = max(int(seconds * hz), 0)
    if count <= 0:
        return
    dt = 1.0 / hz
    for _ in range(count):
        _write_joint_targets(robot, targets, strict=strict)
        time.sleep(dt)


def _jog_selected_joint(
    robot: Any,
    dcss: Any,
    args: argparse.Namespace,
    start_targets: dict[str, list[float]],
    *,
    trace_path: Path,
    strict: bool,
) -> None:
    arm = args.arm
    joint_index = args.joint - 1
    target = {side: list(values) for side, values in start_targets.items()}
    target[arm][joint_index] += float(args.jog_deg)
    dt = 1.0 / float(args.hz)
    read_every = max(int(args.read_every), 1)

    print(
        f"[probe] jog arm={arm} joint={args.joint} "
        f"delta={args.jog_deg:.3f}deg steps={args.steps} hz={args.hz:.3f}"
    )
    for phase_name, begin, end in (
        ("jog_out", start_targets, target),
        ("jog_back", target, start_targets),
    ):
        for index in range(args.steps):
            ratio = (index + 1) / float(args.steps)
            command = {
                side: [
                    float(begin[side][joint])
                    + ratio * (float(end[side][joint]) - float(begin[side][joint]))
                    for joint in range(7)
                ]
                for side in ("A", "B")
            }
            _write_joint_targets(robot, command, strict=strict)
            should_read = index == 0 or index + 1 == args.steps or (index + 1) % read_every == 0
            if should_read:
                feedback = _read_feedback(robot, dcss)
                label = f"{phase_name}_{index + 1:04d}"
                _print_snapshot(label, feedback, target_by_arm=command)
                _append_trace(trace_path, label, feedback, target_by_arm=command)
            time.sleep(dt)


def _write_joint_targets(robot: Any, targets: dict[str, list[float]], *, strict: bool) -> None:
    _write_batch(
        robot,
        ("cmd_A", lambda: robot.set_joint_cmd_pose(arm="A", joints=targets["A"])),
        ("cmd_B", lambda: robot.set_joint_cmd_pose(arm="B", joints=targets["B"])),
        strict=strict,
    )


def _write_batch(
    robot: Any,
    *setters: tuple[str, Callable[[], Any]],
    strict: bool,
) -> None:
    _check_sdk_result("clear_set", robot.clear_set(), strict=strict)
    for name, setter in setters:
        _check_sdk_result(name, setter(), strict=strict)
    _check_sdk_result("send_cmd", robot.send_cmd(), strict=strict)


def _check_sdk_result(name: str, result: Any, *, strict: bool) -> None:
    ok = result is None or result is True
    ok = ok or (isinstance(result, int) and not isinstance(result, bool) and result in (0, 1))
    if ok:
        return
    message = f"[probe] SDK call returned unusual value: {name} -> {result!r}"
    if strict:
        raise RuntimeError(message)
    print(message)


def _read_feedback(robot: Any, dcss: Any) -> dict[str, Any]:
    feedback = robot.subscribe(dcss)
    if not isinstance(feedback, dict):
        raise RuntimeError(f"Unexpected Marvin feedback type: {type(feedback)!r}")
    return feedback


def _current_joints_by_arm(feedback: dict[str, Any]) -> dict[str, list[float]]:
    return {
        "A": _arm_joints(feedback, "A"),
        "B": _arm_joints(feedback, "B"),
    }


def _arm_joints(feedback: dict[str, Any], arm: str) -> list[float]:
    output = _arm_output(feedback, arm)
    joints = [float(value) for value in output.get("fb_joint_pos", [])]
    if len(joints) != 7:
        raise RuntimeError(f"Feedback for arm {arm} did not include 7 joints")
    return joints


def _arm_output(feedback: dict[str, Any], arm: str) -> dict[str, Any]:
    index = 0 if arm == "A" else 1
    outputs = feedback.get("outputs", [])
    if index >= len(outputs):
        raise RuntimeError(f"Feedback does not include arm {arm}")
    output = outputs[index]
    if not isinstance(output, dict):
        raise RuntimeError(f"Feedback output for arm {arm} is not a dict: {type(output)!r}")
    return output


def _arm_state(feedback: dict[str, Any], arm: str) -> dict[str, Any]:
    index = 0 if arm == "A" else 1
    states = feedback.get("states", [])
    if index < len(states) and isinstance(states[index], dict):
        return states[index]
    return {}


def _print_snapshot(
    label: str,
    feedback: dict[str, Any],
    *,
    target_by_arm: dict[str, list[float]] | None = None,
) -> None:
    print(f"[probe] {label}")
    for arm in ("A", "B"):
        output = _arm_output(feedback, arm)
        state = _arm_state(feedback, arm)
        joints = [float(value) for value in output.get("fb_joint_pos", [])][:7]
        cmd = [float(value) for value in output.get("fb_joint_cmd", [])][:7]
        vel = [float(value) for value in output.get("fb_joint_vel", [])][:7]
        print(f"  {arm}_state={state}")
        print(f"  {arm}_joints={_round_list(joints)}")
        if cmd:
            print(f"  {arm}_fb_cmd={_round_list(cmd)}")
        if vel:
            print(f"  {arm}_vel={_round_list(vel)}")
        if target_by_arm is not None and len(joints) == 7:
            err = _max_abs_diff(target_by_arm[arm], joints)
            print(f"  {arm}_target_error_max_deg={err:.4f}")


def _append_trace(
    path: Path,
    label: str,
    feedback: dict[str, Any],
    *,
    target_by_arm: dict[str, list[float]] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "label": label,
        "unix_time": time.time(),
        "arms": {},
    }
    for arm in ("A", "B"):
        output = _arm_output(feedback, arm)
        joints = [float(value) for value in output.get("fb_joint_pos", [])][:7]
        arm_payload: dict[str, Any] = {
            "state": _arm_state(feedback, arm),
            "fb_joint_pos": joints,
            "fb_joint_cmd": [float(value) for value in output.get("fb_joint_cmd", [])][:7],
            "fb_joint_vel": [float(value) for value in output.get("fb_joint_vel", [])][:7],
        }
        if target_by_arm is not None and len(joints) == 7:
            arm_payload["target"] = target_by_arm[arm]
            arm_payload["target_error_max_deg"] = _max_abs_diff(target_by_arm[arm], joints)
        payload["arms"][arm] = arm_payload
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _round_list(values: list[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def _sleep(seconds: float) -> None:
    time.sleep(max(float(seconds), 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
