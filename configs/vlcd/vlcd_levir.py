
_base_ = [
    '../_base_/datasets/base_cd.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py']

import os
data_root = os.path.join(os.environ.get("CDPATH"), 'LEVIR-CD/cut_data')

metainfo = dict(
                classes=('background', 'building'),
                palette=[[0, 0, 0], [255, 255, 255]])

crop_size = (256, 256)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])

norm_cfg = dict(type='SyncBN', requires_grad=True)

find_unused_parameters = True

# VLCD Model Configuration
model = dict(
    type='VLCD',  # Use VLCD instead of ChangeCLIP
    pretrained='pretrained/RN50.pt',
    n_ctx=16,  # Number of learnable context vectors (CoOp)
    context_length=77,
    freeze_clip=True,  # KEY: Freeze CLIP parameters
    text_dim=1024,
    class_names=['background', 'building'],
    
    # CLIP Vision Backbone (Frozen)
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=512,
        style='pytorch'),
    
    # CLIP Text Encoder (Frozen)
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=77,
        embed_dim=1024,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    
    # RS Feature Network (Trainable) - Side Network
    rs_backbone=dict(
        type='RSFeatureNetwork',
        pretrained=True,
        out_indices=(0, 1, 2, 3)),
    
    # Context Decoder (for VL fusion)
    context_decoder=dict(
        type='ContextDecoder',
        context_length=16,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    
    # CFC Module (Change Feature Calculation)
    cfc_module=dict(
        type='CFCModule',
        in_channels=256,
        out_channels=256,
        text_dim=1024,
        num_scales=4),
    
    # FPN Neck
    neck=dict(
        type='FPN',
        in_channels=[256, 256, 256, 256],  # After CFC fusion
        out_channels=256,
        num_outs=4),
    
    # Swin Transformer Decoder with Text
    decode_head=dict(
        type='SwinTextDecode',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_classes=2,
        channels=256,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer with parameter-wise learning rates
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            # Freeze CLIP backbone and text encoder
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'text_encoder': dict(lr_mult=0.0, decay_mult=0.0),
            
            # Train RS backbone (Side Network)
            'rs_backbone': dict(lr_mult=1.0),
            
            # Train Bridging Modules
            'bridging_modules': dict(lr_mult=1.0),
            
            # Train Learnable Prompts (CoOp)
            'learnable_prompt': dict(lr_mult=1.0),
            
            # Train CFC and Decoder
            'cfc': dict(lr_mult=1.0),
            'decode_head': dict(lr_mult=1.0),
            
            # Small lr for normalization layers
            'norm': dict(decay_mult=0.),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.)
        })
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

# Dataset configuration
train_dataloader = dict(
    batch_size=16,  # Slightly smaller batch size due to two backbones
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.txt'))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.txt'))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.txt'))

# Training schedule for 40k iterations
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
