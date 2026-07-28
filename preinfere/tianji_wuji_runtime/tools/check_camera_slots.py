#!/usr/bin/env python3
"""Open configured camera slots once and print RGB frame shapes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from tianji_wuji_runtime.runtime.camera_manager import CameraManager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--camera-capture-fps", type=float, default=60.0)
    args = parser.parse_args()
    keys = [spec.split(":", 1)[0] for spec in args.camera]
    manager = CameraManager.from_cli_specs(
        args.camera,
        required_keys=keys,
        allow_dummy=False,
        width=args.camera_width,
        height=args.camera_height,
        capture_fps=args.camera_capture_fps,
    )
    manager.connect_all()
    try:
        images = manager.read()
        for key, image in images.items():
            print(f"{key}: shape={image.shape}, dtype={image.dtype}")
    finally:
        manager.disconnect_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
