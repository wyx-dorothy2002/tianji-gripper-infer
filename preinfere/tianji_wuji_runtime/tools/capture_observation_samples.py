#!/usr/bin/env python3
"""Capture timestamped observation samples (3 cameras + dual-arm/dual-hand state).

Reads the unified robot state and the latest frame from each of the three camera
streams (``head``, ``left_wrist``, ``right_wrist``), then saves per-sample:

- ``state.json``   : 54-DoF flat state, segmented state, and schema block.
- three PNGs       : ``head.png`` / ``left_wrist.png`` / ``right_wrist.png`` (RGB).
- ``observation.npz``: policy-ready arrays matching the live GR00T input contract.
- ``sample_metadata.json`` + a row in ``samples.jsonl`` with all timestamps.

This tool does NOT touch arm enable / brake / power-on / e-stop flows; those are
owned by the hardware side. It only reads state, grabs frames, and writes to disk.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.camera_manager import CameraManager
from tianji_wuji_runtime.runtime.observation_builder import VIDEO_KEYS, build_policy_observation
from tianji_wuji_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot
from tianji_wuji_runtime.runtime.robot_state import DualArmHandState, ensure_state


def _parse_optional_joint_list(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != schema.LEFT_HAND_DOF:
        raise ValueError(
            f"hand home pose must provide {schema.LEFT_HAND_DOF} comma-separated values, "
            f"got {values.size}"
        )
    return tuple(float(v) for v in values.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-backend", default="fake")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--left-hand-serial", default=None)
    parser.add_argument("--right-hand-serial", default=None)
    parser.add_argument("--left-hand-home", default=None)
    parser.add_argument("--right-hand-home", default=None)
    parser.add_argument("--hand-lowpass-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--tianji-sdk-root", default=None)
    parser.add_argument("--tianji-config-path", default=None)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--image-source", default=None)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=20.0,
        help="Python camera-worker processing frequency.",
    )
    parser.add_argument("--camera-capture-fps", type=float, default=60.0)
    parser.add_argument("--head-stereo-crop", choices=["left", "right"], default=None)
    parser.add_argument("--max-camera-age-ms", type=float, default=150.0)
    parser.add_argument("--camera-warmup-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(RUNTIME_ROOT / "test" / "observation_capture"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.sample_hz <= 0:
        raise ValueError("--sample-hz must be positive")
    if args.camera_fps <= 0:
        raise ValueError("--camera-fps must be positive")
    if args.camera_capture_fps <= 0:
        raise ValueError("--camera-capture-fps must be positive")

    cameras = CameraManager.from_cli_specs(
        args.camera,
        required_keys=VIDEO_KEYS,
        image_source=args.image_source,
        allow_dummy=args.robot_backend == "fake",
        width=args.camera_width,
        height=args.camera_height,
        capture_fps=args.camera_capture_fps,
        fps=args.camera_fps,
        head_stereo_crop=args.head_stereo_crop,
    )
    robot = make_robot(
        RobotConnectionConfig(
            backend=args.robot_backend,
            robot_ip=args.robot_ip,
            left_hand_serial=args.left_hand_serial,
            right_hand_serial=args.right_hand_serial,
            left_hand_home=_parse_optional_joint_list(args.left_hand_home),
            right_hand_home=_parse_optional_joint_list(args.right_hand_home),
            hand_lowpass_cutoff_hz=args.hand_lowpass_cutoff_hz,
            tianji_sdk_root=args.tianji_sdk_root,
            tianji_config_path=args.tianji_config_path,
        )
    )

    run_dir = Path(args.output_dir).expanduser().resolve() / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=False)
    samples_jsonl = run_dir / "samples.jsonl"

    _write_json(run_dir / "metadata.json", _run_metadata(args, cameras))

    dt = 1.0 / float(args.sample_hz)
    robot.connect()
    cameras.connect_all()
    cameras.start_streaming()
    try:
        cameras.wait_until_ready(timeout_sec=args.camera_warmup_sec)
        next_deadline = time.perf_counter()
        for index in range(args.samples):
            now = time.perf_counter()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            record = capture_once(
                robot=robot,
                cameras=cameras,
                task=args.task,
                index=index,
                samples_dir=samples_dir,
                run_dir=run_dir,
                max_camera_age_ms=args.max_camera_age_ms,
            )
            _append_jsonl(samples_jsonl, record)
            cam_ages = ", ".join(
                f"{key}={record['camera'][key]['age_ms']:.1f}ms" for key in VIDEO_KEYS
            )
            print(
                f"[capture] sample {index:04d} "
                f"state_latency={record['state_read_latency_ms']:.2f}ms cam[{cam_ages}]"
            )
            next_deadline += dt
    finally:
        cameras.stop_streaming()
        cameras.disconnect_all()
        robot.disconnect()

    print(f"[capture] saved {args.samples} samples to {run_dir}")
    return 0


def capture_once(
    *,
    robot: Any,
    cameras: CameraManager,
    task: str,
    index: int,
    samples_dir: Path,
    run_dir: Path,
    max_camera_age_ms: float | None,
) -> dict[str, Any]:
    """Read one synchronized-ish observation and persist it. Returns the jsonl row."""
    state_read_start = time.perf_counter()
    state_wall_start = time.time()
    robot_state = ensure_state(robot.get_state())
    state_read_end = time.perf_counter()
    state_wall_end = time.time()
    reference_time = 0.5 * (state_read_start + state_read_end)

    frames = cameras.snapshot_latest(
        reference_time=reference_time,
        max_age_ms=max_camera_age_ms,
    )
    images = {key: frame.image for key, frame in frames.items()}
    observation = build_policy_observation(robot_state, images)

    sample_dir = samples_dir / f"sample_{index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=False)

    paths = {
        "state": sample_dir / "state.json",
        "head": sample_dir / "head.png",
        "left_wrist": sample_dir / "left_wrist.png",
        "right_wrist": sample_dir / "right_wrist.png",
        "observation": sample_dir / "observation.npz",
        "metadata": sample_dir / "sample_metadata.json",
    }

    _write_json(paths["state"], _state_payload(robot_state))
    for key in VIDEO_KEYS:
        Image.fromarray(images[key]).save(paths[key])
    np.savez_compressed(paths["observation"], **observation)

    wall_time = time.time()
    monotonic_time = time.perf_counter()
    camera_meta = {
        key: {
            "frame_id": frame.frame_id,
            "frame_wall_time": frame.wall_time,
            "frame_monotonic_time": frame.monotonic_time,
            "age_ms": (reference_time - frame.monotonic_time) * 1000.0,
            "shape": list(frame.image.shape),
            "dtype": str(frame.image.dtype),
            "source": frame.source,
        }
        for key, frame in frames.items()
    }

    rel_paths = {name: str(path.relative_to(run_dir)) for name, path in paths.items()}
    sample_metadata = {
        "sample_index": index,
        "task": task,
        "wall_time": wall_time,
        "monotonic_time": monotonic_time,
        "state_wall_start": state_wall_start,
        "state_wall_end": state_wall_end,
        "reference_monotonic": reference_time,
        "camera": camera_meta,
        "schema": schema.schema_metadata(),
    }
    _write_json(paths["metadata"], sample_metadata)

    return {
        "sample_index": index,
        "wall_time": wall_time,
        "monotonic_time": monotonic_time,
        "state_read_start_monotonic": state_read_start,
        "state_read_end_monotonic": state_read_end,
        "state_read_latency_ms": (state_read_end - state_read_start) * 1000.0,
        "reference_monotonic": reference_time,
        "paths": rel_paths,
        "camera": camera_meta,
    }


def _state_payload(state: DualArmHandState) -> dict[str, Any]:
    return {
        "flat": state.as_flat().tolist(),
        "segments": state.as_dict(),
        "schema": {
            "state_unit": schema.STATE_UNIT,
            "action_unit": schema.ACTION_UNIT,
            "order": {
                "left_arm": [schema.LEFT_ARM_SLICE.start, schema.LEFT_ARM_SLICE.stop],
                "right_arm": [schema.RIGHT_ARM_SLICE.start, schema.RIGHT_ARM_SLICE.stop],
                "left_hand": [schema.LEFT_HAND_SLICE.start, schema.LEFT_HAND_SLICE.stop],
                "right_hand": [schema.RIGHT_HAND_SLICE.start, schema.RIGHT_HAND_SLICE.stop],
            },
        },
    }


def _run_metadata(args: argparse.Namespace, cameras: CameraManager) -> dict[str, Any]:
    return {
        "created_wall_time": time.time(),
        "created_iso": datetime.now().isoformat(),
        "robot_backend": args.robot_backend,
        "robot_ip": args.robot_ip,
        "left_hand_serial": args.left_hand_serial,
        "right_hand_serial": args.right_hand_serial,
        "camera": [f"{slot.key}:{slot.source}" for slot in cameras.slots],
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "camera_capture_fps": args.camera_capture_fps,
        "camera_processing_fps": args.camera_fps,
        "head_stereo_crop": args.head_stereo_crop,
        "max_camera_age_ms": args.max_camera_age_ms,
        "camera_keys": list(VIDEO_KEYS),
        "samples": args.samples,
        "sample_hz": args.sample_hz,
        "task": args.task,
        "state_unit": schema.STATE_UNIT,
        "schema_order": "left_arm,right_arm,left_hand,right_hand",
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
