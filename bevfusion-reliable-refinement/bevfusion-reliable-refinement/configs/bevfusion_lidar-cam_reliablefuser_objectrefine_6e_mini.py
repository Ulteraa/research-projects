"""nuScenes-mini overlay for reliability-aware fusion + object refinement.

This config is intended for controlled debugging and ablation, not a full
nuScenes benchmark claim. Copy it into projects/BEVFusion/configs/.
"""

_base_ = [
    './bevfusion_lidar-cam_reliablefuser_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

model = dict(
    bbox_head=dict(
        type='ObjectRefineTransFusionHead',
        patch_radius=2,
        refine_num_convs=2,
        refine_weight=1.0,
    ))

# Short controlled experiment used during development.
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        dataset=dict(metainfo=dict(version='v1.0-mini')),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(metainfo=dict(version='v1.0-mini')),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(metainfo=dict(version='v1.0-mini')),
)

work_dir = './work_dirs/bevfusion_lidar-cam_reliablefuser_objectrefine_6e_mini'
