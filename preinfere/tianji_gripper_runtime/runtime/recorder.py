"""Runtime recorder for right-arm gripper policy input/output and traces."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from . import schema
from .action_adapter import ActionAdapter, RightArmGripperAction
from .robot_state import RightArmGripperState


class Recorder:
    def __init__(
        self,
        record_dir: str | Path,
        *,
        config: dict[str, Any],
        adapter: ActionAdapter,
    ) -> None:
        root = Path(record_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "chunks").mkdir(exist_ok=True)
        self.adapter = adapter
        self.chunk_index = 0
        self._write_json(
            self.run_dir / "config.json",
            {
                **_to_jsonable(config),
                "schema": schema.schema_metadata(control_mode=adapter.control_mode),
                "action_adapter": adapter.metadata(),
            },
        )

    def save_chunk(
        self,
        *,
        observation: dict[str, Any],
        raw_chunk: np.ndarray,
        safe_actions: list[RightArmGripperAction],
        safety_events: list[dict[str, object]],
        inference_latency_ms: float | None = None,
    ) -> Path:
        idx = self.chunk_index
        self.chunk_index += 1
        chunk_dir = self.run_dir / "chunks" / f"chunk_{idx:06d}"
        input_dir = chunk_dir / "input"
        output_dir = chunk_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_observation(input_dir, observation)
        np.save(output_dir / "raw_action.npy", np.asarray(raw_chunk, dtype=np.float32))
        safe_chunk = self.adapter.merge_chunk(safe_actions)
        np.save(output_dir / "safe_action.npy", safe_chunk)
        self._write_json(output_dir / "safety_events.json", _to_jsonable(safety_events))
        self._write_json(
            chunk_dir / "metadata.json",
            {
                "chunk_index": idx,
                "timestamp": time.time(),
                "raw_action_shape": list(np.asarray(raw_chunk).shape),
                "safe_action_shape": list(safe_chunk.shape),
                "inference_latency_ms": inference_latency_ms,
            },
        )
        return chunk_dir

    def record_step(
        self,
        *,
        chunk_index: int,
        step_in_chunk: int,
        state_before: RightArmGripperState | np.ndarray | None,
        raw_action: np.ndarray | None,
        safe_action: RightArmGripperAction,
        executed: bool,
        state_after: RightArmGripperState | np.ndarray | None,
        control_latency_ms: float,
        safety_events: list[dict[str, object]] | None = None,
    ) -> None:
        safe_flat = self.adapter.merge_action(safe_action)
        self.append_jsonl(
            self.run_dir / "trajectory.jsonl",
            {
                "timestamp": time.time(),
                "chunk_index": chunk_index,
                "step_in_chunk": step_in_chunk,
                "state_before": _state_to_flat_list(state_before),
                "raw_action": None if raw_action is None else np.asarray(raw_action).tolist(),
                "safe_action": safe_flat.tolist(),
                "executed_action": safe_flat.tolist() if executed else None,
                "state_after": _state_to_flat_list(state_after),
                "control_latency_ms": control_latency_ms,
                "safety_events": safety_events or [],
            },
        )

    def _save_observation(self, input_dir: Path, observation: dict[str, Any]) -> None:
        raw_state = observation.get("state", {})
        if isinstance(raw_state, dict):
            states = {
                key: np.asarray(value).reshape(-1).astype(float).tolist()
                for key, value in raw_state.items()
            }
        else:
            states = {"state": np.asarray(raw_state).reshape(-1).astype(float).tolist()}
        self._write_json(input_dir / "state.json", states)
        language = observation.get("language", {})
        if "prompt" in observation:
            language = {"prompt": observation["prompt"]}
        self._write_json(input_dir / "language.json", _to_jsonable(language))
        for key, value in observation.get("video", {}).items():
            arr = np.asarray(value)
            if arr.ndim == 5:
                frame = arr[0, -1]
            elif arr.ndim == 4:
                frame = arr[-1]
            else:
                continue
            Image.fromarray(frame.astype(np.uint8), mode="RGB").save(
                input_dir / f"{_safe_name(key)}.png"
            )
        for key, value in observation.get("images", {}).items():
            arr = np.asarray(value)
            if arr.ndim != 3:
                continue
            Image.fromarray(arr.astype(np.uint8), mode="RGB").save(
                input_dir / f"{_safe_name(key)}.png"
            )

    @staticmethod
    def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, RightArmGripperState):
        return value.as_flat().tolist()
    if isinstance(value, RightArmGripperAction):
        return value.as_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def _state_to_flat_list(value: RightArmGripperState | np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, RightArmGripperState):
        return value.as_flat().tolist()
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()
