#!/usr/bin/env python3
"""Plot commanded vs actual state from a runtime trajectory.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    with Path(args.trajectory).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("trajectory file is empty")

    commanded = np.asarray([row["safe_action"] for row in rows], dtype=np.float32)
    actual = np.asarray(
        [
            row["state_after"] if row.get("state_after") is not None else row["safe_action"]
            for row in rows
        ],
        dtype=np.float32,
    )

    import matplotlib.pyplot as plt

    output = Path(args.output) if args.output else Path(args.trajectory).with_suffix(".png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    panels = [
        (axes[0], commanded[:, 0:7], actual[:, 0:7], "left_arm"),
        (axes[1], commanded[:, 7:14], actual[:, 7:14], "right_arm"),
        (
            axes[2],
            np.concatenate([commanded[:, 14:34], commanded[:, 34:54]], axis=1),
            np.concatenate([actual[:, 14:34], actual[:, 34:54]], axis=1),
            "hands",
        ),
    ]
    for ax, cmd_panel, actual_panel, title in panels:
        ax.plot(cmd_panel, alpha=0.65)
        ax.plot(actual_panel, linestyle="--", alpha=0.45)
        ax.set_title(title)
        ax.grid(True)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(output)
    print(f"saved plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
