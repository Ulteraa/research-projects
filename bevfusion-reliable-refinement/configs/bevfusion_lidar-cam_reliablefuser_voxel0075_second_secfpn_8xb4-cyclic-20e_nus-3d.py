"""Reliability-aware fusion overlay for the upstream BEVFusion config.

Copy this file into projects/BEVFusion/configs/ in MMDetection3D v1.4.0.
"""

_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

model = dict(
    fusion_layer=dict(
        type='ReliabilityAwareFuser',
        in_channels=[80, 256],
        hidden_channels=128,
        out_channels=256,
    ))

work_dir = './work_dirs/bevfusion_lidar-cam_reliablefuser'
