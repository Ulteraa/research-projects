#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-checkpoints/lift_safe_o2o.pt}"
POLICY="${2:-actor}"
VIDEO="${3:-videos/lift_${POLICY}.mp4}"

python eval_and_record.py \
  --checkpoint "$CHECKPOINT" \
  --video_path "$VIDEO" \
  --num_episodes 50 \
  --max_steps 100 \
  --policy "$POLICY" \
  --env_name Lift \
  --camera_names agentview \
  --image_keys agentview_image \
  --robot_state_keys \
    robot0_eef_pos robot0_eef_quat robot0_gripper_qpos \
    robot0_joint_pos robot0_joint_vel robot0_gripper_qvel object-state
