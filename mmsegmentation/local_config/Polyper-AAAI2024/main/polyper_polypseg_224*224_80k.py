_base_ = [
    '../../_base_/models/swin.py',
    '../../_base_/datasets/polypseg.py',
    '../../default_runtime.py',
    '../../_base_/schedules/schedule_80k_adamw.py'
]

crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)
ham_norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# ham_norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
# find_unused_parameters = True
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),    
    decode_head=dict(
        type='Polyper',
        in_channels=[96,192,384,768],
        channels=96,
        image_size = 128,
        in_index=[0, 1, 2, 3],
        heads= 8,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=6)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader