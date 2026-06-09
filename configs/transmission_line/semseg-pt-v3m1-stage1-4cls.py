_base_ = ["../_base_/default_runtime.py"]

# Stage 1 coarse semantic segmentation for transmission-line scenes.
#
# Original 6 classes:
#   0 ground
#   1 tower
#   2 line
#   3 insulator
#   4 hengdan
#   5 other
#
# Stage-1 4 classes:
#   0 ground
#   1 tower_structure = tower + insulator + hengdan
#   2 line
#   3 other

# Runtime settings
batch_size = 8
num_worker = 8
mix_prob = 0.0
empty_cache = True
enable_amp = True
grad_clip_norm = 1.0

# Model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # XYZ + RGB
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=False,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            # ground / tower_structure / line / other
            # Stage 1 focuses on high recall for tower_structure.
            weight=[0.05, 8.0, 2.0, 0.5],
            loss_weight=1.0,
            ignore_index=-1,
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
)

# Scheduler settings
# First run a sanity check. If results are reasonable, change epoch=100, eval_epoch=10.
epoch = 20
eval_epoch = 5

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# Dataset settings
dataset_type = "S3DISDataset"
data_root = "data/transmission_line_stage1_4cls_balance"
ignore_index = -1
names = ["ground", "tower_structure", "line", "other"]

data = dict(
    num_classes=4,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1 / 12, 1 / 12], axis="z", p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=65536, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=65536, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
            ),
            crop=dict(type="SphereCrop", point_max=100000, mode="all"),
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        ),
    ),
)
