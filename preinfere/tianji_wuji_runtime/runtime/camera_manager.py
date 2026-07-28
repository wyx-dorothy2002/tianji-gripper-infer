"""Camera acquisition, RGB validation, and background streaming.

Two acquisition modes are supported:

- Synchronous ``read()`` (the original contract): read one frame per slot on the
  calling thread. Kept for simple tools and dry-runs.
- Background streaming: ``start_streaming()`` spawns one thread per camera slot
  that keeps the latest frame in a per-slot cache. ``snapshot_latest()`` returns
  the most recent frame of every slot without blocking on a fresh device read.
  This is what the real 20Hz inference loop uses so one slow camera cannot stall
  the others, and so frames keep updating during policy inference / chunk
  execution.

Live devices are configured to capture at the slot target resolution before the
first read. File and dummy sources may still be resized after acquisition.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import numpy as np
from PIL import Image


class CameraError(RuntimeError):
    """Raised when camera acquisition fails."""


_INITIAL_READ_FAILURE_GRACE_SEC = 2.0
_MIN_INITIAL_READ_FAILURES = 3
_CAPTURE_FOURCC = "YUYV"
_NEGOTIATION_TOLERANCE = 0.5
_REALSENSE_SOURCE_PREFIX = "realsense:"


@dataclass
class CameraSlotConfig:
    key: str
    source: str = "dummy"
    width: int = 424
    height: int = 240
    capture_fps: float = 60.0  # Native device stream rate.
    fps: float = 20.0  # Python worker processing rate.
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotate_degrees: int = 0
    stereo_crop: str | None = None  # None, "left", "right"


@dataclass
class LatestFrame:
    """Most recent frame produced by a camera slot's background thread."""

    key: str
    image: np.ndarray
    frame_id: int
    wall_time: float
    monotonic_time: float
    source: str
    width: int
    height: int


class CameraManager:
    def __init__(self, slots: list[CameraSlotConfig]) -> None:
        if not slots:
            raise CameraError("at least one camera slot is required")
        self.slots = slots
        self._captures: dict[str, Any] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._frame_locks: dict[str, threading.Lock] = {}
        self._latest_frames: dict[str, LatestFrame] = {}
        self._thread_errors: dict[str, BaseException] = {}
        self._streaming = False

    @classmethod
    def from_cli_specs(
        cls,
        specs: Iterable[str],
        *,
        required_keys: Iterable[str],
        image_source: str | None = None,
        allow_dummy: bool = False,
        width: int | None = None,
        height: int | None = None,
        capture_fps: float | None = None,
        fps: float | None = None,
        head_stereo_crop: str | None = None,
    ) -> "CameraManager":
        parsed: dict[str, CameraSlotConfig] = {}
        for spec in specs:
            slot = parse_camera_spec(spec)
            parsed[slot.key] = slot

        for key in required_keys:
            if key in parsed:
                continue
            if image_source is not None:
                parsed[key] = CameraSlotConfig(key=key, source=image_source)
            elif allow_dummy:
                parsed[key] = CameraSlotConfig(key=key, source="dummy")
            else:
                raise CameraError(f"missing camera slot {key!r}; pass --camera {key}:<source>")

        for slot in parsed.values():
            if width is not None:
                slot.width = int(width)
            if height is not None:
                slot.height = int(height)
            if capture_fps is not None:
                slot.capture_fps = float(capture_fps)
            if fps is not None:
                slot.fps = float(fps)
            if head_stereo_crop is not None and slot.key == "head":
                slot.stereo_crop = _validate_stereo_crop(head_stereo_crop)

        return cls([parsed[key] for key in parsed])

    # ------------------------------------------------------------------ connect

    def connect_all(self) -> None:
        for slot in self.slots:
            source = _resolve_live_source(slot.source)
            if source in {"dummy", "zeros"} or _looks_like_file(source):
                continue
            if _looks_like_device(source):
                self._captures[slot.key] = self._open_capture(slot, source=source)
            else:
                raise CameraError(
                    f"unsupported camera source {source!r} for {slot.key}; "
                    "use an index, /dev/videoX path, realsense:<serial>, image path, or dummy"
                )

    def disconnect_all(self) -> None:
        if self._streaming:
            self.stop_streaming()
        for capture in self._captures.values():
            try:
                capture.release()
            except Exception:
                pass
        self._captures.clear()

    # ---------------------------------------------------------------- streaming

    def start_streaming(self) -> None:
        if self._streaming:
            return
        self._thread_errors.clear()
        self._latest_frames.clear()
        for slot in self.slots:
            self._frame_locks[slot.key] = threading.Lock()
            self._stop_events[slot.key] = threading.Event()
            thread = threading.Thread(
                target=self._camera_worker,
                args=(slot,),
                name=f"camera-{slot.key}",
                daemon=True,
            )
            self._threads[slot.key] = thread
            thread.start()
        self._streaming = True

    def stop_streaming(self) -> None:
        for event in self._stop_events.values():
            event.set()
        for thread in self._threads.values():
            thread.join(timeout=2.0)
        self._threads.clear()
        self._stop_events.clear()
        self._streaming = False

    def is_streaming(self) -> bool:
        return self._streaming

    def wait_until_ready(
        self,
        *,
        timeout_sec: float,
        poll_interval_sec: float = 0.02,
    ) -> dict[str, LatestFrame]:
        """Wait until every streaming camera has published at least one frame."""
        if not self._streaming:
            raise CameraError("wait_until_ready() requires start_streaming() first")
        deadline = time.perf_counter() + max(timeout_sec, 0.0)
        last_error: CameraError | None = None
        while True:
            try:
                return self.snapshot_latest(max_age_ms=None)
            except CameraError as exc:
                last_error = exc
                if "worker failed" in str(exc):
                    raise
            if time.perf_counter() >= deadline:
                message = f"cameras not ready after {timeout_sec:.1f}s"
                if last_error is not None:
                    message = f"{message}: {last_error}"
                raise CameraError(message)
            time.sleep(poll_interval_sec)

    def _camera_worker(self, slot: CameraSlotConfig) -> None:
        stop_event = self._stop_events[slot.key]
        lock = self._frame_locks[slot.key]
        period = 1.0 / slot.fps if slot.fps > 0 else 0.0
        max_initial_failures = max(
            _MIN_INITIAL_READ_FAILURES,
            int(slot.fps * _INITIAL_READ_FAILURE_GRACE_SEC) if slot.fps > 0 else 1,
        )
        consecutive_failures = 0
        frame_id = 0
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                image = self._acquire_frame(slot)
            except BaseException as exc:  # noqa: BLE001 - surface via snapshot_latest
                consecutive_failures += 1
                with lock:
                    has_frame = slot.key in self._latest_frames
                if not has_frame and consecutive_failures >= max_initial_failures:
                    self._thread_errors[slot.key] = exc
                    return
                if period > 0:
                    stop_event.wait(period)
                continue
            consecutive_failures = 0
            frame_id += 1
            frame = LatestFrame(
                key=slot.key,
                image=image,
                frame_id=frame_id,
                wall_time=time.time(),
                monotonic_time=time.perf_counter(),
                source=slot.source,
                width=slot.width,
                height=slot.height,
            )
            with lock:
                self._latest_frames[slot.key] = frame
            if period > 0:
                remaining = period - (time.perf_counter() - loop_start)
                if remaining > 0:
                    stop_event.wait(remaining)

    def snapshot_latest(
        self,
        reference_time: float | None = None,
        *,
        max_age_ms: float | None = None,
    ) -> dict[str, LatestFrame]:
        if not self._streaming:
            raise CameraError("snapshot_latest() requires start_streaming() first")
        reference = time.perf_counter() if reference_time is None else reference_time
        frames: dict[str, LatestFrame] = {}
        for slot in self.slots:
            key = slot.key
            error = self._thread_errors.get(key)
            if error is not None:
                raise CameraError(f"camera {key} worker failed: {error}") from error
            with self._frame_locks[key]:
                frame = self._latest_frames.get(key)
            if frame is None:
                raise CameraError(f"camera {key} has no frame yet")
            if max_age_ms is not None:
                age_ms = (reference - frame.monotonic_time) * 1000.0
                if age_ms > max_age_ms:
                    raise CameraError(
                        f"camera {key} frame is stale: age {age_ms:.1f}ms > max {max_age_ms:.1f}ms"
                    )
            frames[key] = frame
        return frames

    # ------------------------------------------------------------------- read()

    def read(self) -> dict[str, np.ndarray]:
        """Synchronous one-shot read of every slot.

        When streaming is active this returns the latest cached frame images so we
        never call ``capture.read()`` from two threads at once. Otherwise it reads
        each device synchronously on the calling thread.
        """
        if self._streaming:
            frames = self.snapshot_latest()
            return {key: frame.image for key, frame in frames.items()}
        return {slot.key: self._acquire_frame(slot) for slot in self.slots}

    # ---------------------------------------------------------------- internals

    def _open_capture(self, slot: CameraSlotConfig, *, source: str | None = None) -> Any:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise CameraError(
                "opencv-python is required for live camera devices; "
                "use --image-source for dry-run or install the runtime env"
            ) from exc
        target_source = slot.source if source is None else source
        target = int(target_source) if target_source.isdigit() else target_source
        capture = cv2.VideoCapture(target, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise CameraError(f"failed to open camera {slot.key}:{target_source}")
        try:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*_CAPTURE_FOURCC),
            )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, slot.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, slot.height)
            capture.set(cv2.CAP_PROP_FPS, slot.capture_fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            _validate_capture_negotiation(capture, slot, cv2)
        except Exception:
            capture.release()
            raise
        backend_name = getattr(capture, "getBackendName", lambda: "V4L2")()
        print(
            f"[camera] {slot.key}:{target_source} backend={backend_name} "
            f"native={slot.width}x{slot.height}@{slot.capture_fps:g}fps "
            f"processing={slot.fps:g}fps"
        )
        return capture

    def _acquire_frame(self, slot: CameraSlotConfig) -> np.ndarray:
        image = self._acquire_raw(slot)
        image = _apply_stereo_crop(image, slot)
        image = _apply_orientation(image, slot)
        image = _resize_rgb(image, slot.width, slot.height)
        _validate_rgb(image, slot.key)
        return image

    def _acquire_raw(self, slot: CameraSlotConfig) -> np.ndarray:
        source = slot.source
        if source in {"dummy", "zeros"}:
            return np.zeros((slot.height, slot.width, 3), dtype=np.uint8)
        if _looks_like_file(source):
            path = Path(source)
            if not path.exists():
                raise CameraError(f"image source does not exist for {slot.key}: {path}")
            return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        capture = self._captures.get(slot.key)
        if capture is None:
            raise CameraError(f"camera {slot.key} is not connected")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraError(f"camera {slot.key} read timeout/failure")
        if frame.shape[:2] != (slot.height, slot.width):
            raise CameraError(
                f"camera {slot.key} returned {frame.shape[1]}x{frame.shape[0]}, "
                f"expected native {slot.width}x{slot.height}; refusing post-capture resize"
            )
        # OpenCV returns BGR.
        return np.ascontiguousarray(frame[..., ::-1]).astype(np.uint8)


def parse_camera_spec(spec: str) -> CameraSlotConfig:
    if ":" not in spec:
        raise CameraError(f"camera spec must be key:source, got {spec!r}")
    key, source = spec.split(":", 1)
    key = key.strip()
    source = source.strip()
    if not key or not source:
        raise CameraError(f"camera spec must be key:source, got {spec!r}")
    return CameraSlotConfig(key=key, source=source)


def _validate_stereo_crop(value: str) -> str:
    crop = value.strip().lower()
    if crop not in {"left", "right"}:
        raise CameraError(f"stereo_crop must be 'left' or 'right', got {value!r}")
    return crop


def _looks_like_file(source: str) -> bool:
    suffix = Path(source).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _looks_like_device(source: str) -> bool:
    return source.isdigit() or source.startswith("/dev/video")


def _resolve_live_source(source: str) -> str:
    if source.startswith(_REALSENSE_SOURCE_PREFIX):
        serial = source[len(_REALSENSE_SOURCE_PREFIX) :].strip()
        if not serial:
            raise CameraError("realsense camera source must be realsense:<serial>")
        return _resolve_realsense_video_node(serial)
    return source


def _resolve_realsense_video_node(serial_number: str) -> str:
    devices = _enumerate_realsense_devices()
    device = next((item for item in devices if item.serial_number == serial_number), None)
    if device is None:
        available = ", ".join(sorted(item.serial_number for item in devices)) or "none"
        raise CameraError(
            f"RealSense serial {serial_number} not found. Available serials: {available}"
        )

    physical_port = Path(device.physical_port)
    usb_device_dir = physical_port.parents[2]
    video_nodes = sorted(
        {
            Path("/dev") / path.name
            for path in usb_device_dir.rglob("video*")
            if path.name.startswith("video") and path.name[5:].isdigit()
        },
        key=lambda path: path.name,
    )
    if not video_nodes:
        raise CameraError(
            f"RealSense serial {serial_number} has no video nodes under {usb_device_dir}"
        )

    preferred = [
        path
        for path in video_nodes
        if _node_supports_capture_format(path, fourcc=_CAPTURE_FOURCC)
    ]
    if not preferred:
        available = ", ".join(path.name for path in video_nodes)
        raise CameraError(
            f"RealSense serial {serial_number} has no node advertising {_CAPTURE_FOURCC}; "
            f"found nodes: {available}"
        )
    return str(preferred[-1])


@dataclass(frozen=True)
class _RealSenseDevice:
    serial_number: str
    physical_port: str


def _enumerate_realsense_devices() -> list[_RealSenseDevice]:
    try:
        output = subprocess.check_output(
            ["rs-enumerate-devices"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise CameraError(
            "rs-enumerate-devices is not installed; cannot resolve realsense:<serial>"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise CameraError(f"rs-enumerate-devices failed: {exc.output.strip()}") from exc

    devices: list[_RealSenseDevice] = []
    for block in output.split("Device info:"):
        if "Serial Number" not in block or "Physical Port" not in block:
            continue
        serial_match = re.search(r"Serial Number\s*:\s*(\S+)", block)
        port_match = re.search(r"Physical Port\s*:\s*(\S+)", block)
        if serial_match is None or port_match is None:
            continue
        devices.append(
            _RealSenseDevice(
                serial_number=serial_match.group(1).strip(),
                physical_port=port_match.group(1).strip(),
            )
        )
    return devices


def _node_supports_capture_format(path: Path, *, fourcc: str) -> bool:
    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "-d", str(path), "--list-formats-ext"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return False
    return f"'{fourcc}'" in output


def _apply_stereo_crop(image: np.ndarray, slot: CameraSlotConfig) -> np.ndarray:
    if not slot.stereo_crop:
        return image
    half = image.shape[1] // 2
    if half <= 0:
        raise CameraError(f"camera {slot.key} too narrow to stereo-crop: {image.shape}")
    if slot.stereo_crop == "left":
        return np.ascontiguousarray(image[:, :half])
    return np.ascontiguousarray(image[:, half:])


def _apply_orientation(image: np.ndarray, slot: CameraSlotConfig) -> np.ndarray:
    out = image
    if slot.flip_horizontal:
        out = np.flip(out, axis=1)
    if slot.flip_vertical:
        out = np.flip(out, axis=0)
    if slot.rotate_degrees:
        turns = (slot.rotate_degrees // 90) % 4
        out = np.rot90(out, k=turns)
    return np.ascontiguousarray(out)


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return np.ascontiguousarray(image)
    try:
        import cv2  # type: ignore

        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized.astype(np.uint8))
    except Exception:
        pil = Image.fromarray(image.astype(np.uint8)).resize((width, height), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)


def _validate_capture_negotiation(capture: Any, slot: CameraSlotConfig, cv2: Any) -> None:
    actual_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    mismatches = []
    if abs(actual_width - slot.width) > _NEGOTIATION_TOLERANCE:
        mismatches.append(f"width={actual_width:g}, requested={slot.width}")
    if abs(actual_height - slot.height) > _NEGOTIATION_TOLERANCE:
        mismatches.append(f"height={actual_height:g}, requested={slot.height}")
    if abs(actual_fps - slot.capture_fps) > _NEGOTIATION_TOLERANCE:
        mismatches.append(f"fps={actual_fps:g}, requested={slot.capture_fps:g}")
    if mismatches:
        raise CameraError(
            f"camera {slot.key}:{slot.source} rejected native capture settings: "
            + "; ".join(mismatches)
        )


def _validate_rgb(image: np.ndarray, key: str) -> None:
    if not isinstance(image, np.ndarray):
        raise CameraError(f"camera {key} did not return ndarray")
    if image.dtype != np.uint8:
        raise CameraError(f"camera {key} must return uint8 RGB, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise CameraError(f"camera {key} must return HxWx3 RGB, got {image.shape}")
