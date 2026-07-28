# Tianji Right-Arm Gripper Runtime

This runtime can talk to the original GR00T policy server or an openpi/pi0.5 websocket
policy server. The pi0.5 checkpoint in
`/mnt/efs_1/ziyi/preinfere/openpi_inference_bottle_ckpt20000.tar.gz` is documented by
`openpi/src/openpi/policies/ziyi_policy.py` as:

```text
state/action: right_arm(7) + left_arm(7) + right_gripper(1)
video: head + left_wrist + right_wrist
language: prompt
```

The robot may still be a dual-arm platform. During `connect()`, the runtime reads the left arm
and left gripper initial positions and keeps sending those freeze targets while the policy controls
only the right side in `right_arm_right_gripper` mode.

## Start pi0.5 Policy Server

Extract the openpi tarball somewhere outside this runtime, then start the websocket server:

```bash
cd /path/to/extracted/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ziyi_dualarm \
  --policy.dir /path/to/checkpoint/20000
```

Use the actual config name and checkpoint directory from the extracted training bundle if they
differ. This runtime only needs the server endpoint.

## Control Mode

The default config uses:

```yaml
policy_type: pi
control_mode: right_arm_right_gripper
pi_action_layout: ziyi_15d_right_left_right_gripper
```

In this mode the policy still receives the full 15D training state, but the executor sends only:

```text
right_arm = actions[0:7]
right_gripper = actions[14:15]
```

`dual_arm_dual_gripper` is exposed as a command-line choice, but this checkpoint does not contain
a left-gripper action. The runtime refuses that mode for the ziyi 15D layout instead of sending an
ambiguous command.

## Start GR00T Policy Server

Use the existing GR00T server launcher with the checkpoint trained from
`demo_data/shuiping2026_6_16_lerobot/gripper_n1d7_config.py`:

```bash
cd /home/user/GR00T/tianji_wuji_runtime
CKPT=/path/to/checkpoint EMBODIMENT_TAG=NEW_EMBODIMENT CUDA_VISIBLE_DEVICES=0 \
bash deployment/start_groot_server.sh
```

## Dry Run

The default config is safe: `robot_backend: fake`, `gripper_backend: fake`, and `dry_run: true`.

```bash
cd /home/user/GR00T
uv run python tianji_gripper_runtime/deployment/infer_tianji_gripper.py \
  --config tianji_gripper_runtime/configs/infer.yaml
```

## Hardware Switches

For the Tianji arms, set:

```yaml
robot_backend: tianji_gripper_host
dry_run: false
```

For the gripper, the SDK boundary is
`tianji_gripper_runtime/runtime/gripper_interface.py`.

Until the vendor SDK shape is known, keep:

```yaml
left_gripper_backend: fake
right_gripper_backend: fake
```

Once the right gripper SDK is available, wire it through either:

```yaml
right_gripper_backend: generic_sdk
gripper_sdk_module: your_sdk_module
gripper_sdk_class: YourGripperClass
```

or implement a small `GripperInterface` subclass if the vendor API needs custom handling.

## Launch From This Repo

```bash
cd /home/user/workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ziyi_dualarm \
  --policy.dir /home/user/workspace/openpi/ckpts/blue_bottle/30000
```

Open another terminal for right-arm right-gripper inference:

Use `uv run python` here. A plain `python` may resolve to the active Conda environment, where
the websocket client dependencies are not installed.

```bash
cd /home/user/workspace/openpi
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run python preinfere/tianji_gripper_runtime/deployment/infer_tianji_gripper.py \
  --config preinfere/tianji_gripper_runtime/configs/infer.yaml \
  --policy-type pi \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --control-mode right_arm_right_gripper \
  --task "pick and place the bottle"
```

解压并启动 pi0.5 server
mkdir -p /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle
tar -xzf /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle_ckpt20000.tar.gz \
  -C /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle

cd /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ziyi_dualarm \
  --policy.dir /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle/openpi/checkpoints/pi05_ziyi_dualarm/20000

另开一个终端启动右臂右夹爪推理
cd /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle/openpi
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run python /mnt/efs_1/ziyi/preinfere/tianji_gripper_runtime/deployment/infer_tianji_gripper.py \
  --config /mnt/efs_1/ziyi/preinfere/tianji_gripper_runtime/configs/infer.yaml \
  --policy-type pi \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --control-mode right_arm_right_gripper \
  --task "pick and place the bottle"
