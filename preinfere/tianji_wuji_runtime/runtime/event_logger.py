"""Non-blocking JSONL event logger for runtime timing probes."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import time
from typing import Any

from .recorder import Recorder


class RuntimeEventLogger:
    """Queue events from control threads and write them on a background thread."""

    def __init__(self, path: str | Path, *, max_queue_size: int = 10000) -> None:
        self.path = Path(path)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue_size)
        self._dropped_count = 0
        self._dropped_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="runtime-event-logger", daemon=True)
        self._thread.start()

    def log(self, event: str, **payload: object) -> None:
        if self._closed:
            return
        record: dict[str, Any] = {
            "timestamp": time.time(),
            "monotonic_time": time.perf_counter(),
            "event": event,
            **payload,
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._dropped_lock:
                self._dropped_count += 1

    def close(self, *, timeout_sec: float = 2.0) -> None:
        if self._closed:
            return
        with self._dropped_lock:
            dropped_count = self._dropped_count
        if dropped_count:
            self.log("runtime_event_logger_dropped", dropped_count=dropped_count)
        self._closed = True
        try:
            self._queue.put(None, timeout=timeout_sec)
        except queue.Full:
            # If the queue is full, the daemon thread may still flush what it has.
            return
        self._thread.join(timeout=timeout_sec)

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            if record is None:
                self._queue.task_done()
                break
            try:
                Recorder.append_jsonl(self.path, record)
            finally:
                self._queue.task_done()
