#!/usr/bin/env python3
"""Inspect GR00T server modality config for the loaded checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from tianji_wuji_runtime.runtime.groot_policy_client import GrootPolicyClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    args = parser.parse_args()
    client = GrootPolicyClient(args.policy_host, args.policy_port)
    if not client.ping():
        raise RuntimeError("policy server ping failed")
    configs = client.get_modality_config()
    for name, cfg in configs.items():
        print(f"{name}:")
        print(f"  keys: {list(cfg.modality_keys)}")
        print(f"  delta_indices: {list(cfg.delta_indices)}")
        if hasattr(cfg, "action_configs") and cfg.action_configs:
            print(f"  action_configs: {cfg.action_configs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

