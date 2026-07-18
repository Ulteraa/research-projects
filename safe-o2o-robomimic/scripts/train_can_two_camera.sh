#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-$HOME/robomimic_datasets/can/ph/image_two_cam.hdf5}"
CHECKPOINT="${2:-checkpoints/can_two_camera_safe_o2o.pt}"

python safe_o2o_robomimic.py \
  --dataset "$DATASET" \
  --image_keys agentview_image robot0_eye_in_hand_image \
  --robot_state_keys \
    robot0_eef_pos robot0_eef_quat robot0_gripper_qpos \
    robot0_joint_pos robot0_joint_vel robot0_gripper_qvel object \
  --batch_size 64 \
  --bc_epochs 40 \
  --rl_epochs 10 \
  --num_workers 4 \
  --freeze_rl_encoder \
  --checkpoint_path "$CHECKPOINT"
