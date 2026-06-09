_base_ = ["../_base_/default_runtime.py"]

# Runtime settings.
# A transformed tile with about 57k points used about 6.4 GiB in a forward/backward
# check on the local RTX 3090. Use one tile per step first because dense tiles are
# capped at 100k points below; batch_size=2 can exceed memory for two dense tiles.
batch_size = 1  # total batch size over all GPUs
# Two workers overlap disk reads without putting excessive pressure on a 6.9 GiB dataset.
num_worker = 2
# Do not concatenate scenes: each tile already covers a meaningful corridor area.
mix_prob = 0.0
empty_cache = False
# Full training uses FP32: the first AMP run developed intermittent NaN losses near
# lr=0.00194 before failing inside a half-precision CUDA kernel.
enable_amp = False
# Clip unusually large updates during recovery from the numerically unstable AMP run.
grad_clip_norm = 1.0

# Model settings: every LAS point receives one of the six labels listed under `names`.
model = dict(
    type="DefaultSegmentorV2",
    num_classes=6,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        # Coordinates carry conductor/tower geometry; RGB distinguishes objects
        # with similar shapes. Collect below explicitly builds [coord, color].
        in_channels=6,  # XYZ and RGB
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        # Retain context for long thin lines; memory is controlled by point_max and
        # batch size rather than reducing receptive field before a baseline exists.
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        # Keep this baseline lightweight; test RPE after label mapping is validated.
        # This separates pipeline errors from optional model refinements.
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=False,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
    ),
    criteria=[
        # Ground is about 174.7M of 188.6M retained train points; insulators are
        # below 0.8M. Reduce ground and boost insulators so common background does
        # not dominate cross-entropy optimization.
        dict(
            type="CrossEntropyLoss",
            weight=[0.2, 1.0, 1.0, 2.0, 1.0, 1.0],
            loss_weight=1.0,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Scheduler settings. The initial max_lr=0.002 AMP run became numerically unstable
# after its first saved checkpoint; use a lower FP32 peak for the recovery run.
epoch = 100
# Pointcept runs 10 evaluation cycles, each covering 10 data passes.
# This validates periodically without evaluating after every expensive pass.
eval_epoch = 10
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

# Dataset settings. S3DISDataset reads generic coord/color/semantic_gt `.pth`
# records, which is exactly the schema emitted by the LAS converter.
dataset_type = "S3DISDataset"
data_root = "data/transmission_line"
ignore_index = -1
names = ["ground", "tower", "line", "insulator", "hengdan", "other"]

data = dict(
    num_classes=6,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # Learn local structure rather than absolute map placement.
            dict(type="CenterShift", apply_z=True),
            # Corridors have direction, so keep rotation/scale perturbations modest.
            dict(type="RandomRotate", angle=[-1 / 12, 1 / 12], axis="z", p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # Match preprocessing resolution and limit redundant dense ground samples.
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            # Bound attention memory; raise this only after monitoring peak VRAM.
            dict(type="SphereCrop", point_max=100000, mode="random"),
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
            # Avoid random validation augmentation so metrics remain interpretable.
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=100000, mode="center"),
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
        # Held-out scenes receive deterministic transforms and full-fragment coverage.
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
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
            # One identity pass controls evaluation cost on corridor-scale point clouds.
            aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        ),
    ),
)
