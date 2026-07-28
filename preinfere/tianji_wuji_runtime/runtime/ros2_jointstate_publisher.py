"""Optional ROS 2 JointState publisher for runtime plotting.

The runtime usually runs inside a Python 3.10 conda env, while ROS 2 Jazzy's
``rclpy`` is built for Ubuntu 24.04's system Python 3.12.  To avoid that ABI
split, this wrapper streams JSON lines to a tiny system-Python ROS bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .action_adapter import DualArmHandAction
from .robot_state import DualArmHandState


@dataclass
class Ros2JointStatePublisher:
    topic_prefix: str = "/gr00t_runtime"
    enabled: bool = True

    def __post_init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._warned = False
        if self.enabled:
            self._start_bridge()

    def publish(
        self,
        *,
        target_action: DualArmHandAction,
        current_state: DualArmHandState | None,
    ) -> None:
        if not self.enabled or current_state is None:
            return
        if self._process is None or self._process.poll() is not None:
            self._start_bridge()
            if self._process is None or self._process.poll() is not None:
                return

        payload = {
            "target": target_action.left_arm_q.tolist() + target_action.right_arm_q.tolist(),
            "position": current_state.left_arm_q.tolist() + current_state.right_arm_q.tolist(),
        }
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"[ros2-jointstate] bridge write failed, disabling publisher: {exc}")
            self.enabled = False
            self.close()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                process.kill()

    def _start_bridge(self) -> None:
        bridge_path = Path(__file__).with_name("ros2_jointstate_bridge.py")
        if not bridge_path.exists():
            self._warn_once(f"[ros2-jointstate] bridge script missing: {bridge_path}")
            self.enabled = False
            return

        env = os.environ.copy()
        env.setdefault("ROS_DOMAIN_ID", "0")
        command = (
            "source /opt/ros/jazzy/setup.bash && "
            f"/usr/bin/python3 {shlex.quote(str(bridge_path))} "
            f"--topic-prefix {shlex.quote(self.topic_prefix)}"
        )
        try:
            self._process = subprocess.Popen(
                ["bash", "-lc", command],
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            print(
                "[ros2-jointstate] bridge started with system Python, publishing "
                f"{self.topic_prefix.rstrip('/')}/target and {self.topic_prefix.rstrip('/')}/position"
            )
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"[ros2-jointstate] failed to start bridge: {exc}")
            self.enabled = False

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(message, file=sys.stderr)
