#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-checkpoints/can_two_camera_safe_o2o.pt}"
POLICY="${2:-bc}"
VIDEO="${3:-videos/can_two_camera_${POLICY}.mp4}"

python eval_and_record.py \
  --checkpoint "$CHECKPOINT" \
  --video_path "$VIDEO" \
  --num_episodes 20 \
  --max_steps 150 \
  --policy "$POLICY" \
  --env_name PickPlaceCan \
  --camera_names agentview robot0_eye_in_hand \
  --image_keys agentview_image robot0_eye_in_hand_image \
  --robot_state_keys \
    robot0_eef_pos robot0_eef_quat robot0_gripper_qpos \
    robot0_joint_pos robot0_joint_vel robot0_gripper_qvel object-state
