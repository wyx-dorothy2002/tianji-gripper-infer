#!/usr/bin/env python3
"""Check local LeRobot dataset metadata against Tianji runtime schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from tianji_wuji_runtime.runtime import schema


EXPECTED_SEGMENTS = {
    "left_arm_joint": schema.LEFT_ARM_SLICE,
    "right_arm_joint": schema.RIGHT_ARM_SLICE,
    "left_hand": schema.LEFT_HAND_SLICE,
    "right_hand": schema.RIGHT_HAND_SLICE,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default="/mnt/data/qdhe/workspace/datasets/local_lerobot_dataset",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset_path)
    modality = _load_json(dataset / "meta" / "modality.json")
    info = _load_json(dataset / "meta" / "info.json")
    tasks = _load_jsonl(dataset / "meta" / "tasks.jsonl")

    errors: list[str] = []
    _check_segments("state", modality, errors)
    _check_segments("action", modality, errors)
    _check_feature_dim("observation.state", info, schema.STATE_DIM, errors)
    _check_feature_dim("action", info, schema.ACTION_DIM, errors)
    _check_feature_names(info, errors)
    _check_cameras(modality, info, errors)
    _check_language(modality, info, tasks, errors)

    print(f"dataset: {dataset}")
    print(f"robot_type: {info.get('robot_type')}")
    print(f"fps: {info.get('fps')}")
    print(f"episodes: {info.get('total_episodes')}, frames: {info.get('total_frames')}")
    print(f"runtime schema: {schema.ACTION_SCHEMA_VERSION}")
    print("state/action order:")
    for key, slc in EXPECTED_SEGMENTS.items():
        print(f"  {key}: [{slc.start}, {slc.stop})")
    print("camera keys:")
    for key in modality["video"]:
        feature = info["features"][modality["video"][key]["original_key"]]
        print(f"  {key}: shape={feature['shape']}")
    print("tasks:")
    for task in tasks:
        print(f"  {task.get('task_index')}: {task.get('task')}")

    if errors:
        print("\nFAILED alignment checks:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nOK: dataset metadata is aligned with tianji_wuji_runtime schema")
    return 0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _check_segments(modality_name: str, modality: dict, errors: list[str]) -> None:
    found = modality.get(modality_name, {})
    if list(found.keys()) != list(EXPECTED_SEGMENTS.keys()):
        errors.append(
            f"{modality_name} keys {list(found.keys())} != expected {list(EXPECTED_SEGMENTS.keys())}"
        )
    for key, slc in EXPECTED_SEGMENTS.items():
        entry = found.get(key)
        if entry is None:
            errors.append(f"{modality_name}.{key} missing")
            continue
        if entry.get("start") != slc.start or entry.get("end") != slc.stop:
            errors.append(
                f"{modality_name}.{key} slice [{entry.get('start')}, {entry.get('end')}) "
                f"!= [{slc.start}, {slc.stop})"
            )


def _check_feature_dim(feature_key: str, info: dict, dim: int, errors: list[str]) -> None:
    feature = info.get("features", {}).get(feature_key)
    if feature is None:
        errors.append(f"features.{feature_key} missing")
        return
    if feature.get("shape") != [dim]:
        errors.append(f"features.{feature_key}.shape {feature.get('shape')} != [{dim}]")


def _check_feature_names(info: dict, errors: list[str]) -> None:
    for feature_key in ("observation.state", "action"):
        names = info["features"][feature_key].get("names")
        expected = schema.STATE_KEYS if feature_key == "observation.state" else schema.ACTION_KEYS
        if names != expected:
            errors.append(f"features.{feature_key}.names do not match runtime detailed keys")


def _check_cameras(modality: dict, info: dict, errors: list[str]) -> None:
    expected = ["head", "left_wrist", "right_wrist"]
    video = modality.get("video", {})
    if list(video.keys()) != expected:
        errors.append(f"video keys {list(video.keys())} != expected {expected}")
    for key in expected:
        entry = video.get(key)
        if entry is None:
            continue
        original_key = entry.get("original_key")
        if original_key not in info.get("features", {}):
            errors.append(f"video.{key}.original_key {original_key!r} not in info features")


def _check_language(modality: dict, info: dict, tasks: list[dict], errors: list[str]) -> None:
    annotation = modality.get("annotation", {})
    key = "human.action.task_description"
    if key not in annotation:
        errors.append(f"annotation.{key} missing")
    original_key = annotation.get(key, {}).get("original_key")
    if original_key not in info.get("features", {}):
        errors.append(f"annotation original_key {original_key!r} not in info features")
    if not tasks:
        errors.append("tasks.jsonl is empty")


if __name__ == "__main__":
    raise SystemExit(main())

