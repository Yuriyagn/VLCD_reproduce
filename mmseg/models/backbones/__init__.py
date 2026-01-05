# Copyright (c) OpenMMLab. All rights reserved.
# 精简版 - 仅包含 ChangeCLIP 和 VLCD 所需的骨干网络

from .clip_backbone import (
    CLIPResNet,
    CLIPResNetWithAttention,
    CLIPVisionTransformer,
    CLIPTextEncoder,
    CLIPTextContextEncoder,
    ContextDecoder
)

from .side_fusion_network import (
    SideFusionCLIP,
    RSFeatureNetwork,
    BridgingModule
)

from .side_fusion_vit import (
    SideFusionViT,
    ViTFeatureNetwork,
    ViTBridgingModule
)

__all__ = [
    'CLIPResNet', 'CLIPResNetWithAttention', 'CLIPVisionTransformer',
    'CLIPTextEncoder', 'CLIPTextContextEncoder', 'ContextDecoder',
    'SideFusionCLIP', 'RSFeatureNetwork', 'BridgingModule',
    'SideFusionViT', 'ViTFeatureNetwork', 'ViTBridgingModule'
]
