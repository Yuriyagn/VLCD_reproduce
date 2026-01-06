
_base_ = [
    '../_base_/datasets/base_cd.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py']
import os
# data_root = os.path.join(os.environ.get("CDPATH"), 'Gloucester-SAR')
data_root = '/home/yr/code/CD/Data/Gloucester-SAR'

metainfo = dict(
                classes=('background', 'wetland'),
                palette=[[0, 0, 0], [255, 255, 255]])

crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    ##
    # # 2. 斑点噪声（最重要！）
    # dict(
    #     type='MultiImgSARSpeckleNoise',
    #     noise_var=0.1,      # 噪声强度：0.05(弱) ~ 0.2(强)
    #     prob=0.6            # 60%概率应用
    # ),
    
    # # 3. 高斯噪声
    # dict(
    #     type='MultiImgGaussianNoise',
    #     mean=0,
    #     std=12,             # 标准差
    #     prob=0.4
    # ),
    
    # # 4. Gamma调整（对比度）
    # dict(
    #     type='MultiImgAdjustGamma2',
    #     gamma_range=(0.7, 1.5),  # Gamma范围
    #     prob=0.5
    # ),
    
    # # 5. CLAHE（可选，计算开销大）
    # dict(
    #     type='MultiImgCLAHE2',
    #     clip_limit=2.0,
    #     tile_grid_size=(8, 8),
    #     prob=0.3
    # ),
    # ======================================

    # ========== 几何增强（原有）==========
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),  # 可选：交换时序
    # ==================================
    
    # # ========== 光度学增强（原有）==========
    # dict(
    #     type='MultiImgPhotoMetricDistortion',
    #     brightness_delta=10,
    #     contrast_range=(0.8, 1.2),
    #     saturation_range=(0.8, 1.2),
    #     hue_delta=10
    # ),
    # # ====================================
    
    # # ========== CutMix（新增）==========
    # dict(
    #     type='MultiImgCutMix',
    #     alpha=1.0,
    #     prob=0.2            # 20%概率
    # ),
    # # =================================
    # dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),),
        # saturation_range=(0.8, 1.2),
        # hue_delta=10),
    dict(type='ConcatCDInput'),
    dict(type='PackCDInputs')
]
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
# VLCD ViT Model Configuration
model = dict(
    type='VLCD',
    pretrained='pretrained/ViT-B-16.pt',  # Use ViT-B-16 pretrained weights
    n_ctx=16,                              # Number of CoOp context vectors
    context_length=77,
    freeze_clip=True,                      # Freeze CLIP parameters
    text_dim=512,                          # ViT-B-16 text dimension
    class_names=['background', 'building'],
    
    # CLIP Vision Transformer (Frozen)
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,                         # ViT-B width
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=256,
        out_indices=[3, 5, 7, 11],         # Output 4 stages
        style='pytorch'),
    
    # CLIP Text Encoder (Frozen)
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=77,
        embed_dim=512,                     # ViT text embedding dim
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    
    # RS ViT Feature Network (Trainable) - Side Network
    rs_backbone=dict(
        type='ViTFeatureNetwork',
        pretrained=True,
        embed_dim=768,                     # Match CLIP ViT width
        out_indices=(3, 5, 7, 11),
        img_size=256,
        patch_size=16),
    
    # Context Decoder (for VL fusion)
    context_decoder=dict(
        type='ContextDecoder',
        context_length=16,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,                    # ViT output dim
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    
    # CFC Module (Change Feature Calculation) - ViT version
    cfc_module=dict(
        type='CFCModule',
        in_channels=768,                   # ViT feature dim (after FPN processing)
        out_channels=256,
        text_dim=512,                      # ViT text dim
        num_scales=4),
    
    # FPN Neck - adapted for ViT features
    # After CFC: concat of [F1, F2, diff, rel] = 4 * 768 -> 256
    neck=dict(
        type='FPN',
        in_channels=[256, 256, 256, 256],  # After CFC fusion to 256
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
        text_dim=512,                      # ViT text dim
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#     # optimizer=dict(
#     #     type='AdamW', lr=0.00003, betas=(0.9, 0.999), weight_decay=0.01),
#     accumulative_counts=12,  # 累积4个batch再更新
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
)

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=5000,
#         by_epoch=False,
#     )
# ]

train_dataloader = dict(
    batch_size=4,
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

# training schedule for 20k
# 1. 延长训练到 40k iters
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=20000, 
    val_interval=1000
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=3000, end=20000, by_epoch=False)
]

# 2. 增加学习率（因为模型更复杂了）
# optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, save_best='mIoU', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))